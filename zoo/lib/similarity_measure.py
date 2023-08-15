import logging

import keras
import tensorflow as tf
from keras.layers import Dense

from zoo.util.common_utils import calc_cosine_similarity
from zoo.zoo_constants import ZooConstants


class SimilarityMeasure:
    def __init__(self, config):
        self._config = config

        self._multi_head_num = self._config.get(ZooConstants.MULTI_HEAD_NUM, 1)
        self._similarity_measure = self._config.get(ZooConstants.SIMILARITY_MEASURE, 'mlp')

        self._representation_dim = self._config.get(ZooConstants.SSL_REPRESENTATION_DIM, 128)

        if self._similarity_measure != 'multi_head':
            self._multi_head_num = 1
        else:
            self._query_K_linear = Dense(self._representation_dim)
            self._query_V_linear = Dense(self._representation_dim)
            self._item_K_linear = Dense(self._representation_dim * self._multi_head_num)
            self._item_V_linear = Dense(self._representation_dim * self._multi_head_num)
        logging.info('similarity_measure: {}, multi_head_num: {}'.format(self._similarity_measure, self._multi_head_num))

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
        kernel_size = self._representation_dim()
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

    def calc_similarity(self, query_representation, item_representation):
        if self._similarity_measure == 'mlp':
            qi_score, similarity_auxiliary_loss, show_metrics = self.calc_similarity_mlp(query_representation,
                                                                                         item_representation)
        elif self._similarity_measure == 'cosine':
            qi_score, similarity_auxiliary_loss, show_metrics = self.calc_similarity_cosine(query_representation,
                                                                                            item_representation)
        elif self._similarity_measure == 'kernel':
            qi_score, similarity_auxiliary_loss, show_metrics = self.calc_similarity_kernel(query_representation,
                                                                                            item_representation)
        elif self._similarity_measure == 'multi_head':
            qi_score, similarity_auxiliary_loss, show_metrics = self.calc_similarity_multi_head(query_representation,
                                                                                                item_representation)
        else:
            raise Exception('unsupported similarity measure: {}'.format(self._similarity_measure))

        return qi_score, similarity_auxiliary_loss, show_metrics
