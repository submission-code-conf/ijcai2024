import tensorflow as tf


class TaskBase(object):
    def __init__(self, config):
        super(TaskBase, self).__init__()
        self._config = config

    def assemble_model(self, non_amount_input_dict, amount_input, label, max_amount_input, mode=tf.estimator.ModeKeys.TRAIN):
        """

        Args:
            non_amount_input_dict ():
            amount_input ():
            label ():
            max_amount_input ():

        Returns (loss, output_dict):

        """
        pass

    def metrics_to_show(self):
        return []

    def metrics_to_outptut(self):
        return []
