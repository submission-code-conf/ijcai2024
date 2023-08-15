import tensorflow as tf

from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.zoo_constants import ZooConstants


class PointwiseLTRSubTask(SubTaskBase):
    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        rank_score_logit = task_info_dict.get(self._config.get(ZooConstants.LOGIT_COL), None)
        assert rank_score_logit is not None
        return {}, tf.nn.sigmoid(rank_score_logit)

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        if self._config.get(ZooConstants.SKIP_CALC_LOSS, False):
            return tf.constant(0, dtype=tf.float32), {}

        if mode == tf.estimator.ModeKeys.PREDICT:
            rank_label = self.get_default_label(label_input_dict);
        else:
            rank_label = label_input_dict.get(self._config.get(ZooConstants.RANK_LABEL_COL), None)
            assert rank_label is not None, 'label_input_dict: {}, rank_label: {}'.format(label_input_dict, rank_label)

        rank_score_logit = task_info_dict.get(self._config.get(ZooConstants.LOGIT_COL_LOSS), None)
        assert rank_score_logit is not None

        sample_weights = self.get_sample_weight(label_input_dict)
        rank_loss = tf.compat.v1.losses.log_loss(rank_label, tf.nn.sigmoid(rank_score_logit), weights=tf.reshape(sample_weights, [-1]))

        loss_detail_dict = {
            'rank_label': rank_label,
            'rank_score_logit': rank_score_logit,
            'rank_loss': rank_loss
        }
        return rank_loss, loss_detail_dict

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['rank_label', 'rank_score_logit', 'rank_loss']






