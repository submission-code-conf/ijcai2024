import tensorflow as tf

from zoo.lib.similarity_measure import SimilarityMeasure
from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.util.common_utils import calc_cosine_similarity
from zoo.zoo_constants import ZooConstants


class SimpleDecompositionSimilaritySubTask(SubTaskBase):
    ITEM_SIDE_REPR_COL = 'item_side_repr_col'
    QUERY_SIDE_REPR_COL = 'query_side_repr_col'
    QUERY_ATOM_REPR_COL = 'query_atom_repr_col'
    ITEM_ATOM_REPR_COL = 'item_atom_repr_col'
    QUERY_ATOM_REPR_ALIGN_ORTH_COL = 'query_atom_repr_align_orthogonal_col'
    QUERY_SIDE_REPR_ALIGN_COL = 'query_side_repr_align_col'
    ITEM_ATOM_REPR_ALIGN_ORTH_COL = 'item_atom_repr_align_orthogonal_col'
    ITEM_SIDE_REPR_ALIGN_COL = 'item_side_repr_align_col'

    QUERY_SIDE_ATOM_REPR_SIM_COL = 'query_side_atom_repr_sim_col'
    ITEM_SIDE_ATOM_REPR_SIM_COL = 'item_side_atom_repr_sim_col'

    COSINE_SIM_SCALE_TAU = 'cosine_sim_scale_tau'
    IPW_COSINE_SIM_SCALE_TAU = "ipw_cosine_sim_scale_tau"

    def __init__(self, config):
        super().__init__(config)
        self._similarity_measure = SimilarityMeasure(self._config)

        self._query_atom_repr_col = self._config.get(SimpleDecompositionSimilaritySubTask.QUERY_ATOM_REPR_COL, None)
        self._query_side_repr_col = self._config.get(SimpleDecompositionSimilaritySubTask.QUERY_SIDE_REPR_COL, None)
        self._item_atom_repr_col = self._config.get(SimpleDecompositionSimilaritySubTask.ITEM_ATOM_REPR_COL, None)
        self._item_side_repr_col = self._config.get(SimpleDecompositionSimilaritySubTask.ITEM_SIDE_REPR_COL, None)
        self._query_atom_repr_align_orthogonal_col = self._config.get(SimpleDecompositionSimilaritySubTask.QUERY_ATOM_REPR_ALIGN_ORTH_COL, None)
        self._query_side_repr_align_col = self._config.get(SimpleDecompositionSimilaritySubTask.QUERY_SIDE_REPR_ALIGN_COL, None)
        self._item_atom_repr_align_orthogonal_col = self._config.get(SimpleDecompositionSimilaritySubTask.ITEM_ATOM_REPR_ALIGN_ORTH_COL, None)
        self._item_side_repr_align_col = self._config.get(SimpleDecompositionSimilaritySubTask.ITEM_SIDE_REPR_ALIGN_COL, None)

        self._cosine_sim_scale_tau = self._config.get(SimpleDecompositionSimilaritySubTask.COSINE_SIM_SCALE_TAU, 1.0)
        self._ipw_cosine_sim_scale_tau = self._config.get(SimpleDecompositionSimilaritySubTask.IPW_COSINE_SIM_SCALE_TAU, 1.0)

        self._hidden_units_list = self._config.get(ZooConstants.HIDDEN_UNITS_LIST, [128, 32])

        # self._mlp_atom_orthogonal = keras.Sequential(
        #     [Dense(self._hidden_units_list[i], activation='relu') for i in range(len(self._hidden_units_list) - 1)]
        #         + [Dense(self._hidden_units_list[-1])])
        #
        # self._mlp_side = keras.Sequential(
        #     [Dense(self._hidden_units_list[i], activation='relu') for i in range(len(self._hidden_units_list) - 1)]
        #     + [Dense(self._hidden_units_list[-1])])

    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        query_atom_repr, query_side_repr, item_atom_repr, item_side_repr, query_atom_repr_align_orthogonal, query_side_repr_align, item_atom_repr_align_orthogonal, item_side_repr_align = self._get_reprs(task_info_dict)

        side_ipw_like_weight = self._calc_ipw_like_weight(query_atom_repr_align_orthogonal, query_side_repr_align, item_atom_repr_align_orthogonal, item_side_repr_align)

        # add mlp transform to enhance capability
        # query_atom_repr_orthogonal = self._mlp_atom_orthogonal(query_atom_repr_orthogonal)
        # item_atom_repr_orthogonal = self._mlp_atom_orthogonal(item_atom_repr_orthogonal)
        # query_side_repr = self._mlp_side(query_side_repr)
        # item_side_repr = self._mlp_side(item_side_repr)

        qi_score_logit_atom = calc_cosine_similarity(query_atom_repr, item_atom_repr) / self._cosine_sim_scale_tau
        qi_score_atom = tf.sigmoid(qi_score_logit_atom)

        qi_score_logit_side = calc_cosine_similarity(query_side_repr, item_side_repr) / self._cosine_sim_scale_tau
        qi_score_side = tf.sigmoid(qi_score_logit_side)
        # v1
        qi_score_logit = calc_cosine_similarity(tf.concat([query_atom_repr, query_side_repr], axis=1)
                                                , tf.concat([item_atom_repr, item_side_repr], axis=1)) / self._cosine_sim_scale_tau

        # # v2 - 2023.1.9 11:34 - not good
        # qi_score_logit = calc_cosine_similarity(query_atom_repr_orthogonal + query_side_repr
        #                                         , item_atom_repr_orthogonal + item_side_repr) / self._cosine_sim_scale_tau

        qi_score = tf.sigmoid(qi_score_logit)
        task_info = {
                'qi_score_logit_atom': qi_score_logit_atom,
                'qi_score_logit_side': qi_score_logit_side,
                'qi_score_logit': qi_score_logit,
                'qi_score_atom': qi_score_atom,
                'qi_score_side': qi_score_side,
                'qi_score': qi_score,
                'side_ipw_like_weight': side_ipw_like_weight
        }
        return task_info, tf.sigmoid(qi_score_logit)

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        side_ipw_like_weight = self.get_from_task_info_dict(task_info_dict, 'side_ipw_like_weight')
        assert label_input_dict is not None
        label_input_dict['side_ipw_like_weight'] = side_ipw_like_weight
        task_loss = tf.constant(0.0, dtype=tf.float32)
        loss_detail_dict = {}
        return task_loss, loss_detail_dict

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['qi_score_logit_atom', 'qi_score_logit_side', 'qi_score_logit', 'qi_score_atom',
                                               'qi_score_side', 'qi_score', 'side_ipw_like_weight',
                                              ]

    def do_metrics_to_outptut(self):
        return super().do_metrics_to_outptut() + ['qi_score_logit_atom', 'qi_score_logit_side', 'qi_score_logit', 'qi_score_atom',
                                                  'qi_score_side', 'qi_score', 'side_ipw_like_weight',
                                                  ]

    def _get_reprs(self, task_info_dict):
        query_atom_repr = task_info_dict.get(self._query_atom_repr_col, None)
        query_side_repr = task_info_dict.get(self._query_side_repr_col, None)
        item_atom_repr = task_info_dict.get(self._item_atom_repr_col, None)
        item_side_repr = task_info_dict.get(self._item_side_repr_col, None)
        query_atom_repr_align_orthogonal = task_info_dict.get(self._query_atom_repr_align_orthogonal_col, None)
        query_side_repr_align = task_info_dict.get(self._query_side_repr_align_col, None)
        item_atom_repr_align_orthogonal = task_info_dict.get(self._item_atom_repr_align_orthogonal_col, None)
        item_side_repr_align = task_info_dict.get(self._item_side_repr_align_col, None)

        return query_atom_repr, query_side_repr, item_atom_repr, item_side_repr, query_atom_repr_align_orthogonal,query_side_repr_align, item_atom_repr_align_orthogonal, item_side_repr_align

    def _calc_ipw_like_weight(self, query_atom_repr_orthogonal, query_side_repr, item_atom_repr_orthogonal, item_side_repr):
        # query_side_atom_repr_sim = tf.sigmoid(
        #     (0.0 - calc_cosine_similarity(query_atom_repr_orthogonal + query_side_repr, query_side_repr)) / self._ipw_cosine_sim_scale_tau)
        #
        # item_side_atom_repr_sim = tf.sigmoid(
        #     (0.0 - calc_cosine_similarity(item_atom_repr_orthogonal + item_side_repr, item_side_repr)) / self._ipw_cosine_sim_scale_tau)

        query_side_atom_repr_sim = (1.0 - calc_cosine_similarity(query_atom_repr_orthogonal + query_side_repr, query_side_repr)) / 2
        item_side_atom_repr_sim = (1.0 - calc_cosine_similarity(item_atom_repr_orthogonal + item_side_repr, item_side_repr)) / 2

        mask = tf.cast(query_side_atom_repr_sim < item_side_atom_repr_sim, dtype=tf.float32)
        weight = query_side_atom_repr_sim * mask + item_side_atom_repr_sim * (1.0 - mask)
        weight = tf.reshape(weight, [-1])
        weight = (weight - tf.reduce_min(weight)) / (1e-6 + tf.reduce_max(weight))

        return weight


