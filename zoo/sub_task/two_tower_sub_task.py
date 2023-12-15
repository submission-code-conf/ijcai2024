import keras
import tensorflow as tf
from keras.layers import Dense

from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.util.common_utils import calc_cosine_similarity
from zoo.zoo_constants import ZooConstants


class TwoTowerSubTask(SubTaskBase):
    CONCAT_FIRST = "concat_first"
    QUERY_EMB_SIDE = 'query_emb_side'
    QUERY_EMB_ATOM = 'query_emb_atom'
    ITEM_EMB_SIDE = 'item_emb_side'
    ITEM_EMB_ATOM = 'item_emb_atom'

    QUERY_EMB = 'query_emb'
    ITEM_EMB = 'item_emb'

    ITEM_FEATURE_COLS_SIDE = 'item_feature_cols_side'
    ITEM_FEATURE_COLS_ATOM = 'item_feature_cols_atom'

    QUERY_FEATURE_COLS_SIDE = 'query_feature_cols_side'
    QUERY_FEATURE_COLS_ATOM = 'query_feature_cols_atom'

    ITEM_FEATURE_COLS = 'item_feature_cols'
    QUERY_FEATURE_COLS = 'query_feature_cols'

    def __init__(self, config):
        super().__init__(config)
        self._item_feature_cols_side = self._config.get(TwoTowerSubTask.ITEM_FEATURE_COLS_SIDE, [])
        self._item_feature_cols_atom = self._config.get(TwoTowerSubTask.ITEM_FEATURE_COLS_ATOM, [])
        self._item_feature_cols = self._item_feature_cols_atom + self._item_feature_cols_side

        self._query_feature_cols_side = self._config.get(TwoTowerSubTask.QUERY_FEATURE_COLS_SIDE, [])
        self._query_feature_cols_atom = self._config.get(TwoTowerSubTask.QUERY_FEATURE_COLS_ATOM, [])
        self._query_feature_cols = self._query_feature_cols_atom + self._query_feature_cols_side

        self._hidden_units_list = self._config.get(ZooConstants.HIDDEN_UNITS_LIST, [128, 32])

        self._mlp_query = keras.Sequential(
            [Dense(self._hidden_units_list[i], activation='relu') for i in range(len(self._hidden_units_list) - 1)]
            + [Dense(self._hidden_units_list[-1])])

        self._mlp_item = keras.Sequential(
            [Dense(self._hidden_units_list[i], activation='relu') for i in range(len(self._hidden_units_list) - 1)]
            + [Dense(self._hidden_units_list[-1])])

        self._mlp_query_side = keras.Sequential(
            [Dense(self._hidden_units_list[i], activation='relu') for i in range(len(self._hidden_units_list) - 1)]
            + [Dense(self._hidden_units_list[-1])])

        self._mlp_item_side = keras.Sequential(
            [Dense(self._hidden_units_list[i], activation='relu') for i in range(len(self._hidden_units_list) - 1)]
            + [Dense(self._hidden_units_list[-1])])

        self._mlp_query_atom = keras.Sequential(
            [Dense(self._hidden_units_list[i], activation='relu') for i in range(len(self._hidden_units_list) - 1)]
            + [Dense(self._hidden_units_list[-1])])

        self._mlp_item_atom = keras.Sequential(
            [Dense(self._hidden_units_list[i], activation='relu') for i in range(len(self._hidden_units_list) - 1)]
            + [Dense(self._hidden_units_list[-1])])

        self._tau = self._config.get(ZooConstants.COSINE_SCALE_TAU, 1.0)

        self._concat_first = self._config.get(TwoTowerSubTask.CONCAT_FIRST, True)

    def process_input(self, spare_feature_embs_dict):
        result = {}

        result[TwoTowerSubTask.ITEM_EMB] = self._concat_embs(spare_feature_embs_dict, self._item_feature_cols)
        result[TwoTowerSubTask.QUERY_EMB] = self._concat_embs(spare_feature_embs_dict, self._query_feature_cols)

        result[TwoTowerSubTask.ITEM_EMB_ATOM] = self._concat_embs(spare_feature_embs_dict, self._item_feature_cols_atom)
        result[TwoTowerSubTask.ITEM_EMB_SIDE] = self._concat_embs(spare_feature_embs_dict, self._item_feature_cols_side)

        result[TwoTowerSubTask.QUERY_EMB_ATOM] = self._concat_embs(spare_feature_embs_dict, self._query_feature_cols_atom)
        result[TwoTowerSubTask.QUERY_EMB_SIDE] = self._concat_embs(spare_feature_embs_dict, self._query_feature_cols_side)
        
        # item_embs_list = [spare_feature_embs_dict[col] for col in self._item_feature_cols]
        # item_emb = tf.concat(item_embs_list, axis=1)
        # result[TwoTowerSubTask.ITEM_EMB] = item_emb
        # 
        # query_embs_list = [spare_feature_embs_dict[col] for col in self._query_feature_cols]
        # query_emb = tf.concat(query_embs_list, axis=1)
        # result[TwoTowerSubTask.QUERY_EMB] = query_emb

        return result

    def _concat_embs(self, spare_feature_embs_dict, cols):
        embs_list = [spare_feature_embs_dict[col] for col in cols]
        emb = tf.concat(embs_list, axis=1)
        return emb
    
    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        input_embedding_dict = {}
        if isinstance(input_embedding, dict):
            input_embedding_dict = input_embedding

        if self._concat_first:
            item_emb = input_embedding_dict.get(TwoTowerSubTask.ITEM_EMB, None)
            item_emb = self._mlp_item(item_emb)

            query_emb = input_embedding_dict.get(TwoTowerSubTask.QUERY_EMB, None)
            query_emb = self._mlp_query(query_emb)
        else:
            item_emb = tf.concat([
                self._mlp_item_side(input_embedding_dict.get(TwoTowerSubTask.ITEM_EMB_SIDE, None))
                , self._mlp_item_atom(input_embedding_dict.get(TwoTowerSubTask.ITEM_EMB_ATOM, None))
                ], axis=-1)

            query_emb = tf.concat([
                self._mlp_query_side(input_embedding_dict.get(TwoTowerSubTask.QUERY_EMB_SIDE, None))
                , self._mlp_query_atom(input_embedding_dict.get(TwoTowerSubTask.QUERY_EMB_ATOM, None))
            ], axis=-1)

        cosine_sim = calc_cosine_similarity(item_emb, query_emb)
        cosine_sim_logit = cosine_sim / self._tau

        task_info = {'item_emb': item_emb
            , 'query_emb': query_emb
            , 'cosine_sim': cosine_sim
            , 'cosine_sim_logit': cosine_sim_logit
                     }
        return task_info, cosine_sim_logit

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        loss = tf.constant(0.0, dtype=tf.float32)
        loss_detail_dict = {}
        return loss, loss_detail_dict

    def do_metrics_to_outptut(self):
        task_metrics = ['cosine_sim', 'cosine_sim_logit']
        return super().do_metrics_to_outptut() + task_metrics

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['cosine_sim', 'cosine_sim_logit', 'query_emb', 'item_emb']
    # def metrics_to_show(self):
    #     return super().metrics_to_show() + ['mlp_loss']

    # def loss_weight(self):
    #     return 0
