import logging

import keras
import tensorflow as tf
from keras.layers import Dense

from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.util.common_utils import calc_cosine_similarity
from zoo.util.in_batch_negative_sampling_utils import generate_negative_sampling_cartesian
from zoo.zoo_constants import ZooConstants


class SSLSubTask(SubTaskBase):
    def __init__(self, config):
        super().__init__(config)

        self._multi_head_num = self._config.get(ZooConstants.MULTI_HEAD_NUM, 1)
        self._similarity_measure = self._config.get(ZooConstants.SIMILARITY_MEASURE, 'mlp')
        if self._similarity_measure != 'multi_head':
            self._multi_head_num = 1
        else:
            self._query_K_linear = Dense(self._get_representation_dim())
            self._query_V_linear = Dense(self._get_representation_dim() )
            self._item_K_linear = Dense(self._get_representation_dim() * self._multi_head_num)
            self._item_V_linear = Dense(self._get_representation_dim() * self._multi_head_num)
        logging.info('similarity_measure: {}, multi_head_num: {}'.format(self._similarity_measure, self._multi_head_num))


    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        query_emb, item_emb, query_embs_augmented, item_embs_augmented = self._get_embs(task_info_dict)
        representation_dim = self._get_representation_dim()

        hidden_units_list = self._config.get(ZooConstants.HIDDEN_UNITS_LIST, None)

        # tf.keras.layers.Activation('linear')
        #
        query_mlp_layers = []
        item_mlp_layers = []

        # if hidden_units_list is None:
        #     hidden_units_list = []
        # query_hidden_units_list = hidden_units_list + [representation_dim]
        # item_hidden_units_list = hidden_units_list + [representation_dim * self._multi_head_num]

        if hidden_units_list is not None and len(hidden_units_list) > 0:
            query_mlp_layers.extend([
                Dense(units=hidden_units, activation='relu') for hidden_units in hidden_units_list
                ])
            item_mlp_layers.extend([
                Dense(units=hidden_units, activation='relu') for hidden_units in hidden_units_list
                ])

        query_mlp_layers.append(Dense(units=representation_dim))
        item_mlp_layers.append(Dense(units=representation_dim))

        query_mlp = keras.Sequential(query_mlp_layers)
        item_mlp = keras.Sequential(item_mlp_layers)

        #
        # if self._similarity_measure == 'multi_head':
        #     query_mlp_layers.append()
        #
        # if hidden_units_list is None or len(hidden_units_list) == 0:
        #     query_mlp = tf.keras.layers.Activation('linear')
        #     item_mlp = tf.keras.layers.Activation('linear')
        # else:
        #     query_mlp = keras.Sequential(
        #         [
        #             Dense(units=hidden_units, activation='relu') for hidden_units in hidden_units_list
        #         ] + [Dense(units=representation_dim)]
        #
        #         )
        #     item_mlp = keras.Sequential(
        #         [
        #             Dense(units=hidden_units, activation='relu') for hidden_units in hidden_units_list
        #         ] + [Dense(units=representation_dim)]
        #     )


        query_representation = query_mlp(query_emb)
        query_representations_augmented = [query_mlp(query_emb_augmented) for query_emb_augmented in
                                           query_embs_augmented]

        item_representation = item_mlp(item_emb)
        item_representations_augmented = [item_mlp(item_emb_augmented) for item_emb_augmented in item_embs_augmented]

        # similarity_measure = self._config.get(ZooConstants.SIMILARITY_MEASURE, 'mlp')
        # logging.info('similarity_measure: {}'.format(similarity_measure))
        #

        qi_score, similarity_auxiliary_loss, show_metrics = self._calc_similarity(query_representation,
                        item_representation, self._similarity_measure)


        task_info = {
            'qi_score': qi_score,
            'query_representation': query_representation,
            'item_representation': item_representation,
            'query_representations_augmented': query_representations_augmented,
            'similarity_auxiliary_loss': similarity_auxiliary_loss,
            'item_representations_augmented': item_representations_augmented

        }
        task_info.update(show_metrics)
        return task_info, qi_score

    def calc_similarity_mlp(self, query_representation, item_representation):
        interaction_mlp = keras.Sequential(
            [
                Dense(units=256, activation='relu'),
                Dense(units=1, activation='sigmoid'),
            ]
        )

        qi_score = interaction_mlp(tf.concat([query_representation, item_representation], axis=-1))
        return qi_score, tf.constant(0, dtype=tf.float32), {}

    def calc_similarity_cosine(self, query_representation, item_representation):
        tau = self._config.get(ZooConstants.CONTRASTIVE_LOSS_TAU, 1.0)
        score = tf.nn.sigmoid(calc_cosine_similarity(
            query_representation, item_representation) / tau)

        return score, tf.constant(0, dtype=tf.float32), {}

    # v1
    # def calc_similarity_multi_head(self, query_representation, item_representation):
    #     item_representation = tf.reshape(item_representation, [-1, self._multi_head_num, self._get_representation_dim()])
    #     # query_representation = tf.expand_dims(query_representation, axis=1)
    #     weight = calc_cosine_similarity(
    #         tf.expand_dims(query_representation, axis=1)
    #         , item_representation)
    #     item_representation = tf.reduce_sum(tf.expand_dims(weight, axis=-1) * item_representation, axis=-2)
    #
    #     tau = self._config.get(ZooConstants.CONTRASTIVE_LOSS_TAU, 1.0)
    #     score = tf.nn.sigmoid(calc_cosine_similarity(
    #         query_representation, item_representation) / tau)
    #
    #     return score, tf.constant(0, dtype=tf.float32), {'item_representation_multi_head': item_representation, 'weight_multi_head': weight}

    # v2
    def calc_similarity_multi_head(self, query_representation, item_representation):
        query_K = self._query_K_linear(query_representation)
        query_V = self._query_K_linear(query_representation)

        item_K = self._item_K_linear(item_representation)
        item_V = self._item_V_linear(item_representation)

        item_K = tf.reshape(item_K, [-1, self._multi_head_num, self._get_representation_dim()])
        item_V = tf.reshape(item_V, [-1, self._multi_head_num, self._get_representation_dim()])

        weight = calc_cosine_similarity(
            tf.expand_dims(query_K, axis=1)
            , item_K)

        query_representation = query_V
        item_representation = tf.reduce_sum(tf.expand_dims(weight, axis=-1) * item_V, axis=-2)

        tau = self._config.get(ZooConstants.CONTRASTIVE_LOSS_TAU, 1.0)
        score = tf.nn.sigmoid(calc_cosine_similarity(
            query_representation, item_representation) / tau)

        return score, tf.constant(0, dtype=tf.float32), {'item_representation_multi_head': item_representation, 'weight_multi_head': weight}



    def calc_similarity_kernel(self, query_representation, item_representation):
        kernel_size = self._get_representation_dim()
        Q_mlp = keras.Sequential(
            [
                Dense(units=256, activation='relu'),
                Dense(units=kernel_size * kernel_size, activation='relu'),
            ]
        )

        Q = Q_mlp(tf.concat([query_representation, item_representation], axis=1))
        Q = tf.reshape(Q, [-1, kernel_size, kernel_size])
        Q = tf.nn.l2_normalize(Q, axis=-1)
        query_projection = tf.reduce_sum(Q * tf.expand_dims(query_representation, axis=1), axis=-1)
        item_projection = tf.reduce_sum(Q * tf.expand_dims(item_representation, axis=1), axis=-1)

        lambda_mlp = keras.Sequential(
            [
                Dense(units=256, activation='relu'),
                Dense(units=kernel_size, activation='softplus'),
            ]
        )

        l = lambda_mlp(tf.concat([query_representation, item_representation], axis=1))

        score = tf.reduce_sum(l * query_projection * item_projection, axis=-1)
        score = tf.nn.sigmoid(score)

        show_metrics = {
            'Q': Q,
            'lambda': l
        }

        l_penalty = tf.reduce_sum(tf.square(l))
        Q_penalty = tf.reduce_sum(tf.matmul(Q, Q, transpose_b=True), [-1, -2]) - kernel_size
        Q_penalty = tf.reduce_sum(tf.square(Q_penalty))

        auxiliary_loss = self._config.get('Q_penalty_weight', 1.0) * Q_penalty + \
                         self._config.get('l_penalty_weight', 0.1) * l_penalty

        return score, auxiliary_loss, show_metrics

    # def calc_similarity_multi_head(self, query_representation, item_representation):
    #     tau = self._config.get(ZooConstants.CONTRASTIVE_LOSS_TAU, 1.0)
    #     score = tf.nn.sigmoid(calc_cosine_similarity(
    #         query_representation, item_representation) / tau)
    #
    #     return score, tf.constant(0, dtype=tf.float32)

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        qi_score = task_info_dict[self.add_prefix('qi_score')]
        log_loss = tf.losses.log_loss(tf.reshape(label_input, [-1]), tf.reshape(qi_score, [-1]), weights=sample_weights)

        query_representations_augmented = task_info_dict[self.add_prefix('query_representations_augmented')]
        item_representations_augmented = task_info_dict[self.add_prefix('item_representations_augmented')]

        query_ssl_loss = self._calc_ssl_loss(query_representations_augmented)
        item_ssl_loss = self._calc_ssl_loss(item_representations_augmented)

        cross_ssl_loss = 0
        for query_representation_augmented in query_representations_augmented:
            for item_representation_augmented in item_representations_augmented:
                cross_ssl_loss += self._calc_cross_ssl_loss(query_representation_augmented,
                                                            item_representation_augmented,
                                                            tf.ones_like(label_input))
                                                            # tf.reshape(label_input, [-1]))

        task_loss = self._config.get(ZooConstants.LOG_LOSS_WEIGHT, 0.0) * log_loss + self._config.get(ZooConstants.SSL_LOSS_WEIGHT, 0.1) * (
                query_ssl_loss + item_ssl_loss) + self._config.get(ZooConstants.CROSS_SSL_LOSS_WEIGHT, 0.1) * cross_ssl_loss

        loss_detail_dict = {
            'query_ssl_loss': query_ssl_loss,
            'item_ssl_loss': item_ssl_loss,
            'log_loss': log_loss,
            'task_loss': task_loss,
            'cross_ssl_loss': cross_ssl_loss
        }
        # if self.is_component():
        #     loss = tf.constant(0, dtype=tf.float32)
        return task_loss, loss_detail_dict

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['cross_ssl_loss', 'qi_score', 'query_ssl_loss', 'item_ssl_loss',
                                               'log_loss', 'task_loss', 'item_representation', 'query_representation',
                                               'lambda', 'Q', 'similarity_auxiliary_loss', 'item_representation_multi_head', 'weight_multi_head']

    def _get_embs(self, task_info_dict):
        query_embs_augmented = task_info_dict.get(self._config.get(ZooConstants.QUERY_EMBS_AUGMENTED_COL), [])
        assert len(query_embs_augmented) > 0
        item_embs_augmented = task_info_dict.get(self._config.get(ZooConstants.ITEM_EMBS_AUGMENTED_COL), [])
        assert len(item_embs_augmented) > 0

        return query_embs_augmented[0], item_embs_augmented[1], query_embs_augmented[1:], item_embs_augmented[1:]

    def _calc_ssl_loss(self, representations_augmented):
        emb_l_expanded, emb_r_expanded, label = generate_negative_sampling_cartesian(representations_augmented[0],
                                                                                     representations_augmented[1],
                                                                                     self._config)
        tau = self._config.get(ZooConstants.CONTRASTIVE_LOSS_TAU, 1.0)
        score = tf.nn.sigmoid(calc_cosine_similarity(
            emb_l_expanded, emb_r_expanded) / tau)

        sample_weight = label + (1.0 - label) / (1.0 + tf.reduce_sum(label))
        loss = tf.losses.log_loss(label, score, weights=sample_weight)
        return loss

    def _calc_cross_ssl_loss(self, emb_query, emb_item, mask):
        mask = tf.reshape(mask, [-1])
        emb_query = tf.boolean_mask(emb_query, mask > 0)
        emb_item = tf.boolean_mask(emb_item, mask > 0)
        emb_l_expanded, emb_r_expanded, label = generate_negative_sampling_cartesian(emb_query, emb_item, self._config)
        tau = self._config.get(ZooConstants.CONTRASTIVE_LOSS_TAU, 1.0)
        score, _, _ = self._calc_similarity(emb_l_expanded, emb_r_expanded, self._similarity_measure)
        # score = tf.nn.sigmoid(similarity / tau)

        sample_weight = label + (1.0 - label) * (1.0 + tf.reduce_sum(label)) / (1.0 + tf.reduce_sum(1.0 - label))
        loss = tf.losses.log_loss(tf.reshape(label, [-1]), tf.reshape(score, [-1]), weights=sample_weight)
        return loss

    def _get_representation_dim(self):
        representation_dim = self._config.get(ZooConstants.SSL_REPRESENTATION_DIM, 128)
        return representation_dim

    # def _multi_head(self, emb):
    def _calc_similarity(self, query_representation, item_representation, similarity_measure):
        if similarity_measure == 'mlp':
            qi_score, similarity_auxiliary_loss, show_metrics = self.calc_similarity_mlp(query_representation,
                                                                                         item_representation)
        elif similarity_measure == 'cosine':
            qi_score, similarity_auxiliary_loss, show_metrics = self.calc_similarity_cosine(query_representation,
                                                                                            item_representation)
        elif similarity_measure == 'kernel':
            qi_score, similarity_auxiliary_loss, show_metrics = self.calc_similarity_kernel(query_representation,
                                                                                            item_representation)
        elif similarity_measure == 'multi_head':
            qi_score, similarity_auxiliary_loss, show_metrics = self.calc_similarity_multi_head(query_representation,
                                                                                                item_representation)
        else:
            raise Exception('unsupported similarity measure: {}'.format(similarity_measure))

        return qi_score, similarity_auxiliary_loss, show_metrics
