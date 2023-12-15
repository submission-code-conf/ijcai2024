import logging
logging.getLogger().addHandler(logging.StreamHandler())

import tensorflow as tf
from keras.layers import Dense

from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.zoo_constants import ZooConstants



class MLPSubTask(SubTaskBase):
    def __init__(self, config):
        super().__init__(config)
        self._hidden_units_list = self._config.get(ZooConstants.HIDDEN_UNITS_LIST, [128, 32])
        self._loss_name = self._config.get(ZooConstants.LOSS_NAME, 'log_loss')

    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        hidden_units_list = self._config.get(ZooConstants.HIDDEN_UNITS_LIST, [128, 32])

        x = input_embedding
        for units in hidden_units_list:
            x = Dense(units, activation='relu')(x)
        score = x
        if self._loss_name == 'log_loss':
            score = Dense(1, activation='sigmoid')(score)
        elif self._loss_name == 'mse':
            score = Dense(1)(score)
        else:
            raise Exception('unsupported loss: {}'.format(self._loss_name))
        task_info = {'score': score}
        return task_info, score

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        pred = tf.reshape(task_info_dict[self.add_prefix('score')], [-1])
        y = tf.reshape(label_input, [-1])

        mask = self.get_mask(label_input_dict)
        mask = tf.reshape(mask, [-1, 1])

        logging.info('mlp-task-do_calc_loss: sample_weights: {}, mask: {}'.format(sample_weights, mask))

        masked_sample_weight = tf.reshape(sample_weights, [-1, 1]) * mask

        print('y: {}, pred:{}, weights: {}'.format(y, pred, masked_sample_weight))
        if self._loss_name == 'log_loss':
            loss = tf.compat.v1.losses.log_loss(y, pred, weights=tf.reshape(masked_sample_weight, [-1]))
        elif self._loss_name == 'mse':
            loss = tf.compat.v1.losses.mean_squared_error(tf.reshape(y, [-1]),  tf.reshape(pred, [-1]), weights=tf.reshape(masked_sample_weight, [-1]))
                # 0.0 * tf.reduce_sum(y) + 0.0 * tf.reduce_sum(pred)
                # tf.compat.v1.losses.mean_squared_error(tf.reshape(y, [-1]),  tf.reshape(pred, [-1]), weights=tf.reshape(masked_sample_weight, [-1]))
            # loss = tf.compat.v1.losses.compute_weighted_loss(
            #     tf.losses.mean_squared_error(y, pred), weights=tf.reshape(masked_sample_weight, [-1]))
            #     # .mean_squared_error(y, pred, weights=tf.reshape(masked_sample_weight, [-1]))
        else:
            raise Exception('unsupported loss: {}'.format(self._loss_name))
        loss_detail_dict = {'mlp_loss': loss}
        # if self.is_component():
        #     loss = tf.constant(0, dtype=tf.float32)
        return loss, loss_detail_dict

    def do_metrics_to_outptut(self):
        task_metrics = ['score']
        return super().do_metrics_to_outptut() + task_metrics

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['mlp_loss']








