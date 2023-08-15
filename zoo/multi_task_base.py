class MultiTaskBase(object):
    def __init__(self, config):
        super(MultiTaskBase, self).__init__()
        self._config = config

    def assemble_model(self, non_amount_input_dict, amount_input, label, max_amount_input):
        """

        Args:
            non_amount_input_dict ():
            amount_input ():
            label ():
            max_amount_input ():

        Returns (loss, output_dict, detail_dict):

        """
        pass

    def metrics_to_show(self):
        return []

    def metrics_to_outptut(self):
        return []
