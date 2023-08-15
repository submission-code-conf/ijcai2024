import keras
import tensorflow as tf
from keras.layers import Dense

from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.zoo_constants import ZooConstants


class MultiClassSubTask(SubTaskBase):
    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        multi_class_label = extra_input_dict.get(ZooConstants.MULTI_CLASS_LABEL_COL, None)
        assert multi_class_label is not None

        multi_class_label_classes = self._config.get(ZooConstants.MULTI_CLASS_LABEL_CLASSES, -1)
        assert multi_class_label_classes > 0

        multi_class_label = tf.string_to_hash_bucket_fast(tf.reshape(multi_class_label, [-1]), multi_class_label_classes)
        softmax_emb_dim = 32
        mlp = keras.Sequential(
            [
                Dense(units=256, activation='relu'),
                Dense(units=softmax_emb_dim),
            ]
        )

        multi_class_softmax_input = mlp(input_embedding)

        batch_size = tf.shape(input_embedding)[0]

        multi_class_softmax_weight = tf.get_variable(shape=[multi_class_label_classes, softmax_emb_dim])
        multi_class_softmax_bias = tf.get_variable(shape=[multi_class_label_classes])

        multi_class_softmax_logit = tf.matmul(multi_class_softmax_input, multi_class_softmax_weight, transpose_b=True) + multi_class_softmax_bias
        multi_class_softmax_score = tf.nn.softmax(multi_class_softmax_logit, axis=-1)

        indices = tf.concat([tf.reshape(tf.range(batch_size), [-1, 1]), tf.reshape(multi_class_label, [-1, 1])], axis=1)
        multi_class_hit_score = tf.gather_nd(multi_class_softmax_score, indices=indices)

        task_info = {'multi_class_label': multi_class_label,
                     'multi_class_softmax_input': multi_class_softmax_input,
                     'multi_class_softmax_weight': multi_class_softmax_weight,
                     'multi_class_softmax_bias': multi_class_softmax_bias,
                     'multi_class_hit_score': multi_class_hit_score,

                     }
        return task_info, multi_class_hit_score

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        multi_class_label = task_info_dict[self.add_prefix('multi_class_label')]
        multi_class_softmax_input = task_info_dict[self.add_prefix('multi_class_softmax_input')]
        multi_class_softmax_weight = task_info_dict[self.add_prefix('multi_class_softmax_weight')]
        multi_class_softmax_bias = task_info_dict[self.add_prefix('multi_class_softmax_bias')]

        multi_class_label_classes = self._config.get(ZooConstants.MULTI_CLASS_LABEL_CLASSES, -1)
        assert multi_class_label_classes > 0

        multi_class_label_classes_sampled = self._config.get(ZooConstants.MULTI_CLASS_LABEL_CLASSES_SAMPLED, -1)
        assert multi_class_label_classes > 0

        sample_loss = tf.nn.sampled_softmax_loss(multi_class_softmax_weight
                                          , multi_class_softmax_bias
                                          , multi_class_label
                                          , multi_class_softmax_input
                                          , num_sampled=multi_class_label_classes_sampled
                                          , num_classes=multi_class_label_classes)

        sample_weight = self.get_sample_weight(label_input_dict)
        loss = tf.losses.compute_weighted_loss(sample_loss, sample_weight)


        loss_detail_dict = {'sampled_softmax_loss': loss,
                            'sample_loss': sample_loss
                            }
        # if self.is_component():
        #     loss = tf.constant(0, dtype=tf.float32)
        return loss, loss_detail_dict

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['multi_class_label', 'multi_class_softmax_input'
            , 'multi_class_hit_score', 'sampled_softmax_loss', 'sample_loss']






