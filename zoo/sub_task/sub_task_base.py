import logging
from abc import ABC

import tensorflow as tf
from keras.layers import Dense
from pyhocon import ConfigFactory

from zoo.zoo_constants import ZooConstants


# from zoo.dipn_utils import handle_monotonic_decreasing


class SubTaskBase(ABC):
    def __init__(self, config):
        super(SubTaskBase, self).__init__()
        if isinstance(config, dict):
            self._config = ConfigFactory.from_dict(config)
        else:
            self._config = config
        # self._config.amount_buckets = tf.convert_to_tensor(sorted(self._config.get(ZooConstants.AMOUNT_BUCKETS, [])),
        #                                                    dtype=tf.float32)
        # self._config.amount_bucket_incs = self._config.amount_buckets[1:] - self._config.amount_buckets[:-1]

        self.as_component = False

    # self._sub_task_config = sub_task_config

    def metrics_to_show(self):
        # return self.add_prefix(self.do_metrics_to_show())
        return list(set(self.add_prefix(self.do_metrics_to_show() + self.metrics_to_outptut())))

    # , self.name() + '-label_sum': label_sum
    # , self.name() + '-label_input': label_input

    def do_metrics_to_show(self):
        return [self.name() + '-mask_sum', self.name() + '-sample_weight_sum', self.name() + '-sample_weights',
                self.name() + '-label_sum', self.name() + '-label_input']

    # self.name() + '-mask_sum': mask_sum
    # ,self.name() + '-sample_weight_sum': sample_weight_sum
    #
    def metrics_to_outptut(self):
        # return self.do_metrics_to_outptut()
        return self.add_prefix(self.do_metrics_to_outptut())

    def do_metrics_to_outptut(self):
        return [self.name()]

    def process_input(self, spare_feature_embs_dict):
        """
        Args:
        Returns:
            dict: intermediate feature name -> tensor
        """
        embs_list = [item[1] for item in sorted(spare_feature_embs_dict.items(), key=lambda t: t[0])]
        if len(embs_list) == 0:
            return None
        emb = tf.concat(embs_list, axis=1)
        return emb

    def sparse_feature_sepc(self):
        """
        Args:
        Returns:
            (prerequisite sparse feature indices as list, is exclusive )
        """
        feature_group_list = self._config.get(ZooConstants.FEATURE_GROUP_LIST, None)
        is_exclusive = self._config.get(ZooConstants.IS_EXCLUSIVE, False)

        return feature_group_list, is_exclusive

    def construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        """
        Args:
        Returns:
            (task dict, prediction)
        """
        inputs = []
        if input_embedding is not None:
            inputs.append(input_embedding)

        input_keys_from_extra = self._config.get(ZooConstants.INPUTS_FROM_EXTRA, [])
        if len(input_keys_from_extra) > 0:
            # align_dense = Dense(16)

            inputs_from_extra = [Dense(16)(extra_input_dict[key]) for key in input_keys_from_extra]
            inputs.extend(inputs_from_extra)

        input_keys_from_task = self._config.get(ZooConstants.INPUTS_FROM_TASK, [])
        if len(input_keys_from_task) > 0:
            inputs_from_task = [task_info_dict[key] for key in input_keys_from_task]
            inputs.extend(inputs_from_task)

        if len(inputs) == 0:
            logging.info(
                'len(inputs) == 0, feature_group_list: {}, input_keys_from_extra: {}, input_keys_from_task: {}'.format(
                    self._config.get(ZooConstants.FEATURE_GROUP_LIST, None)
                    , input_keys_from_extra
                    , input_keys_from_task))

        # assert len(inputs) >= 1,

        if len(inputs) > 1:
            input_embedding = tf.concat(inputs, axis=1)
        elif len(inputs) == 1:
            input_embedding = inputs[0]
        else:
            input_embedding = None

        treatment_key = self._config.get(ZooConstants.TREATMENT, None)
        if treatment_key:
            treatment = extra_input_dict.get(treatment_key)

        if treatment is not None and self._config.get(ZooConstants.MONOTONIC_DECREASING
                , ZooConstants.MONOTONIC_DECREASING_DEFAULT):
            treatment = self._handle_monotonic_decreasing(self._config.get(ZooConstants.MONOTONIC_DECREASING
                                                                           , ZooConstants.MONOTONIC_DECREASING_DEFAULT)
                                                          , treatment
                                                          , self._config.get(ZooConstants.AMOUNT_BUCKETS, [])[-1])

        task_info_dict, prediction = self.do_construct_model(input_embedding, treatment, extra_input_dict,
                                                             task_info_dict)
        task_info_dict = self.add_prefix(task_info_dict)
        return task_info_dict, prediction

    def _handle_monotonic_decreasing(self, monotonic_decreasing, amount_input, amount_range_ub=-1):
        result_amount_input = amount_input
        if monotonic_decreasing:
            assert amount_range_ub >= 0
            if isinstance(amount_input, list):
                result_amount_input = [amount_range_ub - v for v in amount_input]
            else:
                result_amount_input = amount_range_ub - amount_input
        return result_amount_input

    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        """
        Args:
        Returns:
            (task dict, prediction)
        """
        raise NotImplementedError('subclasses must override this method!')

    def calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None, mode=tf.estimator.ModeKeys.TRAIN):
        """
        Args:
        Returns:
            (loss, loss detail dict)
        """

        label_col = self.label_col()
        if label_col:
            assert label_col in label_input_dict.keys(), "label_input_dict.keys(): {}".format(label_input_dict.keys())
            logging.info('{}: use custom label: {}'.format(self.name(), self.label_col()))
            label_input = label_input_dict[label_col]
        #
        # if sample_weights is None:
        #     if label_input_dict and len(label_input_dict) > 0:
        #         sample_weights = self.get_sample_weight(label_input_dict)
        #     else:
        #         sample_weights = self.get_sample_weight(label_input)

        if sample_weights is None:
            sample_weights = self.get_sample_weight(label_input_dict)
        #     else:
        #         sample_weights = self.get_sample_weight(label_input)

        mask = self.get_mask(label_input_dict)
        mask = tf.reshape(mask, [-1, 1])
        mask_sum = tf.reduce_sum(mask)
        sample_weight_sum = tf.reduce_sum(sample_weights)
        sample_weights_0 = sample_weights
        label_sum = tf.reduce_sum(label_input)

        # logging.info('mlp-task-do_calc_loss: sample_weights: {}, mask: {}'.format(sample_weights, mask))

        sample_weights = tf.reshape(tf.reshape(sample_weights, [-1, 1]) * mask, [-1])

        loss, loss_detail_dict = self.do_calc_loss(task_info_dict, label_input, sample_weights, task_name,
                                                   label_input_dict)
        loss_detail_dict.update({
            self.name() + '-mask_sum': mask_sum
            , self.name() + '-sample_weight_sum': sample_weight_sum
            , self.name() + '-sample_weights': sample_weights_0
            , self.name() + '-label_sum': label_sum
            , self.name() + '-label_input': label_input
        })
        loss_detail_dict = self.add_prefix(loss_detail_dict)

        return loss, loss_detail_dict

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        """
        Args:
        Returns:
            (loss, loss detail dict)
            :param mode:
        """
        raise NotImplementedError('subclasses must override this method!')

    def observe_avg(self, amount_input, label_input, task_info_dict):
        """
        Args:
        Returns:
            task dict
        """
        return task_info_dict
        # raise NotImplementedError('subclasses must override this method!')

    def name(self):
        return self._config.get('name', None)

    def is_component(self):
        return self.as_component

    def mask_col(self):
        return self._config.get(ZooConstants.MASK_COL, None)

    def label_col(self):
        return self._config.get(ZooConstants.LABEL_COL, None)

    def sample_weight_col(self):
        return self._config.get(ZooConstants.SAMPLE_WEIGHT_COL, None)

    def get_sample_weight(self, label_input_dict):
        sample_weight_key = self.sample_weight_col()
        if sample_weight_key:
            sample_weight = label_input_dict[sample_weight_key]
        else:
            batch_size = tf.shape(list(label_input_dict.values())[0])[0]
            sample_weight = tf.ones([batch_size, 1])
        logging.info('get_sample_weight, sample_weight_key: {}, sample_weight: {}'.format(sample_weight_key, sample_weight))
        return tf.reshape(sample_weight, [-1])

    def get_mask(self, label_input_dict):
        mask_key = self.mask_col()
        if mask_key:
            mask = label_input_dict[mask_key]
        else:
            batch_size = tf.shape(list(label_input_dict.values())[0])[0]
            mask = tf.ones([batch_size, 1])
        logging.info('get_mask, mask_col: {}, mask: {}'.format(mask_key, mask))
        return mask

    def loss_weight(self):
        return self._config.get(ZooConstants.LOSS_WEIGHT, 1.0)

    def add_prefix(self, l):
        if not self._config.get(ZooConstants.ADD_PREFIX, False):
            return l

        if isinstance(l, list):
            return ['{}_{}'.format(self.name(), v) for v in l]
        elif isinstance(l, dict):
            return dict([('{}_{}'.format(self.name(), k), v) for k, v in l.items()])
        elif isinstance(l, str):
            return '{}_{}'.format(self.name(), l)
        else:
            raise Exception('unsupported type: {}'.format(type(l)))

    def get_input_batch_size(self, label_input_dict):
        inout_batch_size = tf.shape(list(label_input_dict.values())[0])[0]
        logging.info('label_input_dict: {}'.format(label_input_dict))
        if ZooConstants.INPUT_BATCH_SIZE in label_input_dict.keys():
            inout_batch_size = label_input_dict[ZooConstants.INPUT_BATCH_SIZE]
        return inout_batch_size

    def get_current_batch_size(self, label_input_dict):
        inout_batch_size = tf.shape(list(label_input_dict.values())[0])[0]
        return inout_batch_size


    def get_default_label(self, label_input_dict):
        batch_size = self.get_current_batch_size(label_input_dict)
        label = tf.zeros([batch_size], dtype=tf.float32)
        return label

    def get_from_task_info_dict(self, task_info_dick, key):
        return task_info_dick[self.add_prefix(key)]
