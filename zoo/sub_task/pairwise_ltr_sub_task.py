import tensorflow as tf

from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.util.in_batch_negative_sampling_utils import construct_pairwise_sample
from zoo.zoo_constants import ZooConstants


class PairwiseLTRSubTask(SubTaskBase):
    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        rank_score_logit = task_info_dict.get(self._config.get(ZooConstants.LOGIT_COL), None)
        assert rank_score_logit is not None
        return {}, tf.nn.sigmoid(rank_score_logit)

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        if self._config.get(ZooConstants.SKIP_CALC_LOSS, False):
            return tf.constant(0, dtype=tf.float32), {}

        rank_label = label_input_dict.get(self._config.get(ZooConstants.RANK_LABEL_COL), None)
        assert rank_label is not None, 'label_input_dict: {}, rank_label: {}'.format(label_input_dict, rank_label)

        rank_score_logit = task_info_dict.get(self._config.get(ZooConstants.LOGIT_COL), None)
        assert rank_score_logit is not None
        # rank_score_logit = tf.log(rank_score / (1 - rank_score + 1e-8))

        session_id = label_input_dict.get(self._config.get(ZooConstants.SESSION_ID_COL), None)
        assert session_id is not None

        logit_diff, label_diff, _, _ = construct_pairwise_sample(rank_score_logit, rank_label, session_id)

        label_diff = tf.cast(tf.cast(label_diff > 0, dtype=tf.int32), dtype=tf.float32)

        rank_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.reshape(label_diff, shape=[-1])
                , logits=tf.reshape(logit_diff, shape=[-1])))

        loss_detail_dict = {
            'rank_label': rank_label,
            'rank_score_logit': rank_score_logit,
            'label_diff': label_diff,
            'logit_diff': logit_diff,
            'rank_loss': rank_loss,
        }
        return rank_loss, loss_detail_dict

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['rank_label', 'rank_score_logit', 'label_diff', 'logit_diff', 'rank_loss']






