import copy
import logging

import tensorflow as tf
from keras import backend

from zoo.task_base import TaskBase
from zoo.util.sub_task_utils import get_sub_task_class
from zoo.zoo_constants import ZooConstants


class TaskAssembler(TaskBase):
    def __init__(self, config):
        print(config)
        logging.info('MultiTaskAssembler.super().__init__(config): {}'.format(config))
        self._config = config
        self._sub_tasks = self._get_sub_tasks()

    # 输入
    # input_embedding_list：list[tensor], 经过特征处理的embedding list, 年龄：bucket-》 embedding lookup ，地域embedding lookup
    # file：特征 index
    # treatment ： 没有用到
    # label_input_dict: dict[string -> tensor], 样本权重，label value
    # extra_input_dict: dict[string -> embedding]，加载的连续类的tensor
    # 输出：
    # loss ->
    # output_dict -> dict[string -> embedding]
    # locals()
    def assemble_model(self, input_embedding_list, treatment, label_input_dict, extra_input_dict, mode=tf.estimator.ModeKeys.TRAIN):
        config = self._config

        if label_input_dict is None:
            label_input_dict = {}

        label_mapping = self._config.get('label_mapping')

        input_embedding_dict = dict(enumerate(input_embedding_list))

        if self._config.get(ZooConstants.ADD_BATCH_NEGATIVE_SAMPLING, False):
            input_embedding_dict, treatment, label_input_dict, extra_input_dict =\
                self._add_negative_sampling(input_embedding_dict, treatment, label_input_dict, extra_input_dict)

        task_dict = {}
        output_dict = {}
        metric_dict = {}
        sub_task_input_embeddings = []
        for task_name, label_value in label_mapping.items():
            with tf.compat.v1.variable_scope(task_name):
                task_info_dict = {}
                for sub_task in self._sub_tasks:
                    with tf.compat.v1.variable_scope(sub_task.name()):
                        indices, is_exclusive = sub_task.sparse_feature_sepc()
                        spare_feature_embs_dict = {}
                        if indices is None:
                            spare_feature_embs_dict = input_embedding_dict
                        else:
                            for index in indices:
                                if is_exclusive:
                                    spare_feature_embs_dict[index] = input_embedding_dict.pop(index, None)
                                else:
                                    spare_feature_embs_dict[index] = input_embedding_dict.get(index, None)
                        sub_task_input_embedding = sub_task.process_input(spare_feature_embs_dict)
                        sub_task_input_embeddings.append(sub_task_input_embedding)
                        logging.info('sub_task_name: {}, indices: {}, is_exclusive: {}, sub_task_input_embedding: {}'
                                     .format(sub_task.name(), indices, is_exclusive, sub_task_input_embedding))
                        if isinstance(sub_task_input_embedding, dict):
                            extra_input_dict.update(sub_task_input_embedding)
                            # sub_task_input_embedding = None

                        sub_task_info_dict, sub_pred = sub_task.construct_model(sub_task_input_embedding, treatment, extra_input_dict, task_info_dict)
                        # sub_task_info_dict = sub_task.add_prefix(sub_task_info_dict)
                        sub_task_info_dict[sub_task.name()] = sub_pred
                        task_info_dict.update(sub_task_info_dict)

            task_info_dict[task_name] = sub_pred
            output_dict[task_name] = sub_pred
            task_dict[task_name] = task_info_dict

        with tf.name_scope('loss'):
            label_keys = []
            for task_name, task_info_dict in task_dict.items():
                label_keys.append(task_name)
                logging.info('label_input_dict.keys(): {}'.format(label_input_dict.keys()))
                if label_mapping[task_name] in label_input_dict.keys():
                    label_input = label_input_dict[label_mapping[task_name]]
                else:
                    logging.info(str.format('{} not in {}', label_mapping[task_name], label_input_dict.keys()))
                    label_input = tf.zeros([tf.shape(input_embedding_list[0])[0]], dtype=tf.float32)
                    label_input_dict[label_mapping[task_name]] = label_input


                # sample_weights = tf.reshape(extra_input_dict[ZooConstants.SAMPLE_WEIGHT], [-1])

                total_loss = 0
                loss_detail_dict = {}
                for sub_task in self._sub_tasks:
                    sub_task_loss, sub_task_loss_detail_dict = sub_task.calc_loss(task_info_dict, label_input, None, task_name, label_input_dict, mode)
                    total_loss += sub_task_loss * sub_task.loss_weight()
                    logging.info('sub_task:{}, sub_task.loss_weight: {}'.format(sub_task.name(), sub_task.loss_weight()))
                    loss_detail_dict.update(sub_task_loss_detail_dict)

                    # def calc_loss(self, task, label_input, sample_weights, task_name):
                task_info_dict.update(loss_detail_dict)
                task_info_dict[ZooConstants.TOTAL_LOSS] = total_loss

                for sub_task in self._sub_tasks:
                    task_info_dict = sub_task.observe_avg(treatment, label_input, task_info_dict)
                    task_output_dict = {k: task_info_dict[k] for k in sub_task.metrics_to_outptut() if k in task_info_dict.keys()}
                    output_dict.update(task_output_dict)

                    task_metric_dict = {k: task_info_dict[k] for k in sub_task.metrics_to_show() if k in task_info_dict.keys()}
                    metric_dict.update(task_metric_dict)

            task_weight_dict = self._config.get(ZooConstants.TASK_WEIGHT_DICT, ZooConstants.TASK_WEIGHT_DICT_DEFAULT)
            assert len(task_weight_dict) == 0 or len(task_weight_dict) == len(
                set(task_dict.keys()) & set(task_weight_dict.keys())), \
                str.format(
                    'task_dict.keys():{}, task_weight_dict.keys():{}, len(task_weight_dict): {},' +
                    'len(set(task_dict.keys()) & set(task_weight_dict.keys())): {}',
                    task_dict.keys(), task_weight_dict.keys(), len(task_weight_dict),
                    len(set(task_dict.keys()) & set(task_weight_dict.keys())))
            loss = 0.0
            for label_name, task_info_dict in task_dict.items():
                loss += task_info_dict[ZooConstants.TOTAL_LOSS] * task_weight_dict.get(label_name, 1.0)
        return loss, output_dict, metric_dict

    def metrics_to_show(self):
        metrics = []
        for task in self._sub_tasks:
            metrics.extend(task.metrics_to_show())
        return super(TaskAssembler, self).metrics_to_show() \
               + metrics

    def metrics_to_outptut(self):
        metrics = []
        for task in self._sub_tasks:
            metrics.extend(task.metrics_to_outptut())
        return super(TaskAssembler, self).metrics_to_outptut() + metrics

    def _handle_negative_sampling(self, label_key, output_dict, task):
        r = self._config.get(ZooConstants.NEGATIVE_SAMPLING_RATIO, ZooConstants.NEGATIVE_SAMPLING_RATIO_DEFAULT)
        r = self._config.get(ZooConstants.NEGATIVE_SAMPLING_RATIO_MAPPING,
                             ZooConstants.NEGATIVE_SAMPLING_RATIO_MAPPING_DEFAULT).get(label_key, r)
        logging.info(str.format("negative_sampling_ratio: {}", r))
        p = task[label_key]
        calibrated = p / (p + (1 - p) / r)
        task['calibrated'] = calibrated
        return task, tf.constant(0.0)

    def _get_sub_tasks(self):
        sub_tasks = []
        sub_task_names = self._config.get(ZooConstants.SUB_TASK_NAMES, ZooConstants.SUB_TASK_NAMES_DEFAULT)
        as_component = False
        if len(sub_task_names) > 1:
            as_component = True
        for sub_task_name in sub_task_names:
            sub_task_config = {}

            sub_task_config.update(copy.deepcopy(self._config))
            logging.info('_get_sub_tasks, after update, sub_task_name: {}. sub_task_config: {}'.format(sub_task_name, sub_task_config))

            sub_task_config.update(copy.deepcopy(self._config.get(sub_task_name, {})))
            logging.info('_get_sub_tasks, type(self._config): {}, config: {}, sub_task_config: {}'.format(type(self._config), self._config, sub_task_config))

            sub_task_class_name = sub_task_config.get(ZooConstants.SUB_TASK_CLASS_NAME)
            assert sub_task_class_name is not None, 'sub_task_class_name is None {}'.format(sub_task_config)
            sub_task = get_sub_task_class(sub_task_class_name)(sub_task_config)
            sub_task.as_component = as_component
            sub_tasks.append(sub_task)
        return sub_tasks

    def _get_mask_cols(self):
        mask_cols = []
        for sub_task in self._sub_tasks:
            mask_col = sub_task.mask_col()
            if mask_col:
                mask_cols.append(mask_col)
        return mask_cols

    def _add_negative_sampling(self, input_embedding_dict, treatment, label_input_dict, extra_input_dict):
        rand_shuffle_group_list = self._config.get(ZooConstants.RAND_SHUFFLE_GROUP_LIST, [])
        assert len(rand_shuffle_group_list) > 0

        negative_sampling_strategy = self._config.get(ZooConstants.NEGATIVE_SAMPLING_STRATEGY, 'rand')
        if 'rand' == negative_sampling_strategy:
            return self._add_negative_sampling_rand(input_embedding_dict, treatment, label_input_dict, extra_input_dict)
        elif 'cartesian':
            return self._add_negative_sampling_cartesian(input_embedding_dict, treatment, label_input_dict, extra_input_dict)
        else:
            raise Exception('unsupported negative_sampling_strategy: {}'.format(negative_sampling_strategy))

    def _add_negative_sampling_cartesian(self, input_embedding_dict, treatment, label_input_dict, extra_input_dict):
        rand_shuffle_group_list = self._config.get(ZooConstants.RAND_SHUFFLE_GROUP_LIST, [])
        assert len(rand_shuffle_group_list) > 0

        batch_size = tf.shape(list(input_embedding_dict.values())[0])[0]
        rand_indices = tf.random.shuffle(tf.range(batch_size))
        shuffle_indices_negative, repeat_indices_negative = self._generate_indices_v2(batch_size)

        negative_sample_num = tf.shape(shuffle_indices_negative)[0]

        shuffle_indices = tf.concat([tf.range(batch_size), shuffle_indices_negative], axis=0)
        repeat_indices = tf.concat([tf.range(batch_size), repeat_indices_negative], axis=0)

        learning_phase = backend.learning_phase()
        logging.info('learning_phase: {}'.format(learning_phase))
        shuffle_indices = tf.cond(learning_phase, lambda: shuffle_indices, lambda: tf.range(batch_size))
        repeat_indices = tf.cond(learning_phase, lambda: repeat_indices, lambda: tf.range(batch_size))

        input_embedding_dict_extended = {}
        for k, v in input_embedding_dict.items():
            # v_negative = tf.gather(v)
            if k in rand_shuffle_group_list:
                input_embedding_dict_extended[k] = tf.gather(v, shuffle_indices)
            else:
                input_embedding_dict_extended[k] = tf.gather(v, repeat_indices)

        treatment_extended = tf.gather(treatment, shuffle_indices)

        label_input_dict_extended = {}
        mask_cols = self._get_mask_cols()
        for k, v in label_input_dict.items():
            if k not in mask_cols:
                # tf.zeros([, tf.shape(v)[1]])
                v_extended = tf.concat([v, 0 * tf.gather(v, repeat_indices_negative)], axis=0)
            else:
                v_extended = tf.concat([v, tf.gather(v, repeat_indices_negative)], axis=0)
            label_input_dict_extended[k] = tf.cond(
                learning_phase, lambda: v_extended, lambda: v)
        label_input_dict_extended[ZooConstants.IN_BATCH_SAMPLING_MASK] = tf.cond(
            learning_phase, lambda: tf.concat([tf.ones([batch_size]), 0 * tf.ones([negative_sample_num])], axis=0), lambda: tf.ones([batch_size]))

        label_input_dict_extended[ZooConstants.INPUT_BATCH_SIZE] = batch_size

        rand_shuffle_extra_input_features = self._config.get(ZooConstants.RAND_SHUFFLE_EXTRA_INPUT_FEATURES, [])
        extra_input_dict_extended = {}
        for k, v in extra_input_dict.items():
            if k == ZooConstants.SAMPLE_WEIGHT:
                sample_weight = extra_input_dict[ZooConstants.SAMPLE_WEIGHT]
                extra_input_dict_extended[ZooConstants.SAMPLE_WEIGHT] = tf.cond(
                    learning_phase, lambda: tf.concat([tf.ones([batch_size]),
                                                        # (0.0 + tf.cast(batch_size, dtype=tf.float32))
                                                       tf.cast(batch_size, dtype=tf.float32) / tf.cast(negative_sample_num, dtype=tf.float32)
                                                       # / (tf.log(1.0 + tf.cast(negative_sample_num, dtype=tf.float32)))
                                                       * tf.ones([negative_sample_num])]
                                                      , axis=0)
                    , lambda: tf.ones([batch_size]))
                continue
            if k in rand_shuffle_extra_input_features:
                extra_input_dict_extended[k] = tf.gather(v, shuffle_indices)
            else:
                extra_input_dict_extended[k] = tf.gather(v, repeat_indices)
        logging.info('after _add_negative_sampling')

        return input_embedding_dict_extended, treatment_extended, label_input_dict_extended, extra_input_dict_extended

    def _generate_indices_v2(self, batch_size):
        repeats = batch_size
        if self._config.get(ZooConstants.IN_BATCH_NEGATIVE_SAMPLING_REPEATS, -1) > 0:
            repeats = self._config.get(ZooConstants.IN_BATCH_NEGATIVE_SAMPLING_REPEATS, -1)

        repeats = tf.where(tf.greater(repeats, batch_size), batch_size, repeats)

        #  0,0,.0,1,1.1,..,n-1,.,n-1
        repeat_indices = self._cartesian_expand(
                tf.reshape(tf.range(batch_size), [-1, 1])
                , tf.ones([repeats, 1], dtype=tf.int32)
        )

        #  0,1,2,.,n-1,0,1,2,.,n-1,.,0,1,2,.,n-1
        shuffle_indices = self._cartesian_expand(
            tf.ones([batch_size, 1], dtype=tf.int32)
            , tf.reshape(tf.random_shuffle(tf.range(batch_size))[:repeats], [-1, 1])
            )

        mask = tf.not_equal(repeat_indices, shuffle_indices)

        # repeat_indices = tf.where(mask, ,repeat_indices)

        shuffle_indices = tf.where(mask, shuffle_indices, tf.mod(repeat_indices + 1, batch_size))
            # tf.boolean_mask(shuffle_indices, mask)

        return shuffle_indices, repeat_indices

    def _cartesian_expand(self, l, r):
        s = tf.reshape(tf.linalg.matmul(tf.reshape(l, [-1, 1]), tf.reshape(r, [-1, 1]), transpose_b=True), [-1])
        return s

    def _generate_indices(self, batch_size):
        i = tf.ones([batch_size], dtype=tf.int32)
        r = tf.range(batch_size)

        if self._config.get(ZooConstants.IN_BATCH_NEGATIVE_SAMPLING_REPEATS, -1) > 0:
            repeats = self._config.get(ZooConstants.IN_BATCH_NEGATIVE_SAMPLING_REPEATS, -1)
            r = tf.random_shuffle(r)[repeats]
            # r = tf.random.uniform(shape=[repeats], minval=0, maxval=batch_size, dtype=tf.int32)
            logging.info("add_negative_sampling-generate_indices, repeats: {}, r: {}".format(repeats, r))

        s = tf.reshape(tf.linalg.matmul(tf.reshape(i, [-1, 1]), tf.reshape(r, [-1, 1]), transpose_b=True), [-1])
        r = tf.reshape(tf.linalg.matmul(tf.reshape(r, [-1, 1]), tf.reshape(i, [-1, 1]), transpose_b=True), [-1])

        mask = tf.not_equal(r, s)

        r2 = tf.boolean_mask(r, mask)

        s2 = tf.boolean_mask(s, mask)

        return s2, r2


    def _add_negative_sampling_rand(self, input_embedding_dict, treatment, label_input_dict, extra_input_dict):
        rand_shuffle_group_list = self._config.get(ZooConstants.RAND_SHUFFLE_GROUP_LIST, [])
        assert len(rand_shuffle_group_list) > 0

        batch_size = tf.shape(list(input_embedding_dict.values())[0])[0]
        rand_indices = tf.random.shuffle(tf.range(batch_size))
        shuffle_indices = tf.concat([tf.range(batch_size), rand_indices], axis=0)
        repeat_indices = tf.concat([tf.range(batch_size), tf.range(batch_size)], axis=0)

        learning_phase = backend.learning_phase()
        logging.info('learning_phase: {}'.format(learning_phase))
        shuffle_indices = tf.cond(learning_phase, lambda: shuffle_indices, lambda: tf.range(batch_size))
        repeat_indices = tf.cond(learning_phase, lambda: repeat_indices, lambda: tf.range(batch_size))

        input_embedding_dict_extended = {}
        for k, v in input_embedding_dict.items():
            # v_negative = tf.gather(v)
            if k in rand_shuffle_group_list:
                input_embedding_dict_extended[k] = tf.gather(v, shuffle_indices)
            else:
                input_embedding_dict_extended[k] = tf.gather(v, repeat_indices)

        treatment_extended = tf.gather(treatment, shuffle_indices)

        label_input_dict_extended = {}
        mask_cols = self._get_mask_cols()
        for k, v in label_input_dict.items():
            if k not in mask_cols:
                v_extended = tf.concat([v, 0 * v], axis=0)
            else:
                v_extended = tf.concat([v, v], axis=0)
            label_input_dict_extended[k] = tf.cond(
                learning_phase, lambda: v_extended, lambda: v)
        label_input_dict_extended[ZooConstants.IN_BATCH_SAMPLING_MASK] = tf.cond(
            learning_phase, lambda: tf.concat([tf.ones([batch_size]), 0 * tf.ones([batch_size])], axis=0), lambda: tf.ones([batch_size]))
        label_input_dict_extended[ZooConstants.INPUT_BATCH_SIZE] = batch_size

        rand_shuffle_extra_input_features = self._config.get(ZooConstants.RAND_SHUFFLE_EXTRA_INPUT_FEATURES, [])
        extra_input_dict_extended = {}
        for k, v in extra_input_dict.items():
            if k in rand_shuffle_extra_input_features:
                extra_input_dict_extended[k] = tf.gather(v, shuffle_indices)
            else:
                extra_input_dict_extended[k] = tf.gather(v, repeat_indices)
        logging.info('after _add_negative_sampling')

        return input_embedding_dict_extended, treatment_extended, label_input_dict_extended, extra_input_dict_extended

