import logging

import tensorflow as tf
from keras import backend

from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.zoo_constants import ZooConstants


class SSLAugmentSubTask(SubTaskBase):
    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        batch_size = tf.shape(input_embedding)[0]
            # self.get_current_batch_size(extra_input_dict)
        embs_augmented = []
        embs_augmented.append(input_embedding)
        atom_feature_correlation = self._get_atom_feature_correlation(extra_input_dict)
        atom_feature_mask_out_prob = self._config.get(ZooConstants.ATOM_FEATURE_MASK_OUT_PROB, 0)
        mask_out_prob = atom_feature_mask_out_prob * atom_feature_correlation
        matrices = []
        field_embeddings_masked = []
        for i in range(2):
            with tf.compat.v1.variable_scope('augment-{}-{}'.format(self.name(), i)):
                # mask out atom_feature and its highly correlated features by probability
                rnd_matrix = tf.random.uniform([tf.shape(mask_out_prob)[0], tf.shape(mask_out_prob)[1]])
                field_mask = tf.cast(rnd_matrix < mask_out_prob, dtype=tf.float32)

                logging.info('self._config.sparse_embedding_dim: {}'.format(self._config.sparse_embedding_dim))

                field_embedding = tf.reshape(input_embedding, [batch_size, -1, self._config.sparse_embedding_dim])
                field_embedding_masked = (1.0 - tf.expand_dims(field_mask, axis=2)) * field_embedding

                learning_phase = backend.learning_phase()
                logging.info('learning_phase: {}'.format(learning_phase))
                field_embedding = tf.cond(learning_phase, lambda: field_embedding_masked, lambda: field_embedding)

                input_embedding_augmented = tf.reshape(field_embedding, [batch_size, -1])

                # drop out
                dropout_rate = self._config.get(ZooConstants.SSL_AUGMENT_DROPOUT_RATE, 0)
                input_embedding_augmented = tf.keras.layers.Dropout(dropout_rate)(input_embedding_augmented)
                emb_dim = self._config.sparse_embedding_dim * len(self.sparse_feature_sepc()[0])
                embs_augmented.append(tf.reshape(input_embedding_augmented, [-1, emb_dim]))

        task_info = {'input_embedding': input_embedding,
                     'input_embedding_augmented_0': embs_augmented[0],
                     'input_embedding_augmented_1': embs_augmented[1],
                     'embs_augmented': embs_augmented,

                     }
        return task_info, 0.0 * tf.reduce_sum(input_embedding_augmented, axis=1)

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        loss = tf.constant(0.0, dtype=tf.float32)
        loss_detail_dict = {'ssl_augment_loss': loss}
        # if self.is_component():
        #     loss = tf.constant(0, dtype=tf.float32)
        return loss, loss_detail_dict

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['input_embedding', 'input_embedding_augmented_0', 'input_embedding_augmented_1']

    def _get_atom_feature_correlation(self, extra_input_dict):
        atom_feature_correlation = extra_input_dict.get(self._config.get(ZooConstants.ATOM_FEATURE_CORRELATION_COL))
        feature_group_list, _ = self.sparse_feature_sepc()
        # atom_feature_correlation = atom_feature_correlation[:, feature_group_list]

        atom_feature_correlation = tf.transpose(
            tf.gather(tf.transpose(atom_feature_correlation), tf.constant(feature_group_list, dtype=tf.int32)))

        return atom_feature_correlation







