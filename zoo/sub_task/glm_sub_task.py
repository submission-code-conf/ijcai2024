import tensorflow as tf

from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.zoo_constants import ZooConstants


class GLMSubTask(SubTaskBase):
    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        logit = task_info_dict.get(self._config.get(ZooConstants.LOGIT_COL), None)
        assert logit is not None

        score = logit

        task_info = {'score': score,
                     }
        return task_info, score

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        score = task_info_dict[self.add_prefix('score')]
        loss = tf.losses.mean_squared_error(tf.reshape(label_input, [-1])
                                     , tf.reshape(score, [-1]), weights=tf.reshape(sample_weights, [-1]))

        loss_detail_dict = {'glm_loss': loss,
                            }

        return loss, loss_detail_dict

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['score', 'glm_loss']






