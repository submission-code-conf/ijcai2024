import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda

from zoo.lib.similarity_measure import SimilarityMeasure
from zoo.sub_task.sub_task_base import SubTaskBase
from zoo.util.common_utils import calc_cosine_similarity
from zoo.util.common_utils import create_mlp_layer_v2
from zoo.util.in_batch_negative_sampling_utils import generate_negative_sampling_cartesian_2, \
    generate_negative_sampling_cartesian_3
from zoo.zoo_constants import ZooConstants


class EndogenyDecompositionSubTask(SubTaskBase):
    STAGE1_LOSS_TAU = "stage1_loss_tau"
    STAGE1_LOSS_MARGIN = "stage1_loss_margin"
    STAGE1_LOSS_WEIGHT = "stage1_loss_weight"
    ATOM_FEATURE_EMB = 'atom_feature_emb'
    SIDE_FEATURE_EMB = 'side_feature_emb'

    ATOM_FEATURE_COLS = 'atom_feature_cols'
    SIDE_FEATURE_COLS = 'side_feature_cols'
    ALIGNMENT_DIM = 'alignment_dim'
    PROJECTION_DIM = 'projection_dim'

    def __init__(self, config):
        super().__init__(config)

        self._atom_feature_cols = self._config.get(EndogenyDecompositionSubTask.ATOM_FEATURE_COLS, [])
        self._side_feature_cols = self._config.get(EndogenyDecompositionSubTask.SIDE_FEATURE_COLS, [])
        assert len(self._atom_feature_cols) > 0
        assert len(self._side_feature_cols) > 0

        self._alignment_dim = self._config.get(EndogenyDecompositionSubTask.ALIGNMENT_DIM, 128)
        self._projection_dim = self._config.get(EndogenyDecompositionSubTask.PROJECTION_DIM, 128)

        similarity_measure_config_default = {
            ZooConstants.SIMILARITY_MEASURE: 'cosine'
            , ZooConstants.CONTRASTIVE_LOSS_TAU: 0.1}
        self._similarity_measure_config = self._config.get(
            ZooConstants.SIMILARITY_MEASURE_CONFIG, similarity_measure_config_default)

        self._n_repeats = self._config.get(ZooConstants.N_REPEATS, 128)

        self._ssl_loss_weight = self._config.get(ZooConstants.SSL_LOSS_WEIGHT, 0.1)

        self._stage1_loss_weight = self._config.get(EndogenyDecompositionSubTask.STAGE1_LOSS_WEIGHT, 1.0)

        self._stage1_loss_margin = self._config.get(EndogenyDecompositionSubTask.STAGE1_LOSS_MARGIN, 0.5)
        self._stage1_loss_tau = self._config.get(EndogenyDecompositionSubTask.STAGE1_LOSS_TAU, 0.1)

    def process_input(self, spare_feature_embs_dict):
        result = {}
        print('process_input: spare_feature_embs_dict', spare_feature_embs_dict)
        print('process_input: atom_feature_cols, side_feature_cols', self._atom_feature_cols, self._side_feature_cols)

        atom_embs_list = [spare_feature_embs_dict[col] for col in self._atom_feature_cols]
        atom_feature_emb = tf.concat(atom_embs_list, axis=1)
        result[EndogenyDecompositionSubTask.ATOM_FEATURE_EMB] = atom_feature_emb

        side_embs_list = [spare_feature_embs_dict[col] for col in self._side_feature_cols]
        side_feature_emb = tf.concat(side_embs_list, axis=1)
        result[EndogenyDecompositionSubTask.SIDE_FEATURE_EMB] = side_feature_emb
        print('process_input: result', result)

        return result

    def do_construct_model(self, input_embedding, treatment, extra_input_dict, task_info_dict=None):
        print('input_embedding', input_embedding)
        input_embedding_dict = {}
        if isinstance(input_embedding, dict):
            input_embedding_dict = input_embedding

        atom_feature_emb = input_embedding_dict.get(EndogenyDecompositionSubTask.ATOM_FEATURE_EMB, None)
        atom_repr_projection,  atom_repr_alignment = self._create_mlp_and_do_mapping(atom_feature_emb, 'atom')

        side_feature_emb = input_embedding_dict.get(EndogenyDecompositionSubTask.SIDE_FEATURE_EMB, None)
        side_repr_projection, side_repr_alignment = self._create_mlp_and_do_mapping(side_feature_emb, 'side')

        cosine_sim_alignment = calc_cosine_similarity(atom_repr_alignment, side_repr_alignment)
        side_repr_alignment_normalized = tf.nn.l2_normalize(side_repr_alignment, axis=-1)

        atom_repr_alignment_orthogonal = atom_repr_alignment - tf.expand_dims(tf.reduce_sum(atom_repr_alignment * side_repr_alignment_normalized, axis=-1), axis=1) * side_repr_alignment_normalized

        task_info = {'atom_repr': atom_repr_projection,
                     'side_repr': side_repr_projection,
                     'atom_repr_alignment': atom_repr_alignment,
                     'side_repr_alignment': side_repr_alignment,
                     'cosine_sim_alignment': cosine_sim_alignment,
                     'atom_repr_alignment_orthogonal': atom_repr_alignment_orthogonal,
                     }
        return task_info, tf.concat([atom_repr_projection, side_repr_projection], axis=-1)

    def do_calc_loss(self, task_info_dict, label_input, sample_weights, task_name, label_input_dict=None,
                     mode=tf.estimator.ModeKeys.TRAIN):
        side_repr_alignment = self.get_from_task_info_dict(task_info_dict, 'side_repr_alignment')
        atom_repr_alignment_orthogonal = self.get_from_task_info_dict(task_info_dict, 'atom_repr_alignment_orthogonal')

        cosine_sim = self.get_from_task_info_dict(task_info_dict, 'cosine_sim_alignment')
        gt_margin = tf.cast(cosine_sim > self._stage1_loss_margin, tf.float32)

        stage1_loss = tf.reduce_mean((1 - gt_margin) * tf.square((self._stage1_loss_margin - cosine_sim) / self._stage1_loss_tau))

        side_repr_ssl_loss, almost_identity_ratio_side, almost_identity_ratio_loss_side =\
            self._do_calc_ssl_loss(side_repr_alignment)

        atom_repr_ssl_loss, cosine_sim_side, almost_identity_ratio_atom, almost_identity_ratio_loss_side_atom = \
            self._do_calc_ssl_loss_atom(atom_repr_alignment_orthogonal, side_repr_alignment)

        cosine_sim_side_max = tf.reduce_max(cosine_sim_side)
        cosine_sim_side_min = tf.reduce_min(cosine_sim_side)
        cosine_sim_side_mean = tf.reduce_mean(cosine_sim_side)

        loss = self._stage1_loss_weight * stage1_loss + self._ssl_loss_weight * (atom_repr_ssl_loss + side_repr_ssl_loss)

        loss_detail_dict = {'stage1_loss': stage1_loss
                            , 'atom_repr_ssl_loss': atom_repr_ssl_loss
                            , 'side_repr_ssl_loss': side_repr_ssl_loss
                            , 'cosine_sim_side': cosine_sim_side
                            , 'cosine_sim_side_max': cosine_sim_side_max
                            , 'cosine_sim_side_min': cosine_sim_side_min
                            , 'cosine_sim_side_mean': cosine_sim_side_mean
                            , 'almost_identity_ratio_side': almost_identity_ratio_side
                            , 'almost_identity_ratio_loss_side': almost_identity_ratio_loss_side
                            , 'almost_identity_ratio_atom': almost_identity_ratio_atom
                            , 'almost_identity_ratio_loss_side_atom': almost_identity_ratio_loss_side_atom
                            }

        return loss, loss_detail_dict

    def _do_calc_ssl_loss(self, emb):
        emb_expanded, emb_augmented_expanded, label = \
            generate_negative_sampling_cartesian_2(emb, emb, self._n_repeats)
        similarity_measure = SimilarityMeasure(self._similarity_measure_config)
        similarity_score, _, _ = similarity_measure.calc_similarity(emb_expanded, emb_augmented_expanded)

        cosine_sim = (1.0 + calc_cosine_similarity(emb_expanded, emb_augmented_expanded)) / 2
        almost_identity = tf.cast(cosine_sim > self._stage1_loss_margin, dtype=tf.int32)
        almost_identity_ratio = tf.reduce_sum((1.0 - label) * tf.cast(almost_identity, tf.float32)) / tf.reduce_sum(1.0 - label)

        almost_identity_ratio_loss = tf.compat.v1.losses.log_loss(label, similarity_score, weights=(1.0 - label) * tf.cast(almost_identity, tf.float32))

        loss = tf.compat.v1.losses.log_loss(label, similarity_score, weights=1.0 - (1.0 - label) * tf.cast(almost_identity, tf.float32))

        return loss, almost_identity_ratio, almost_identity_ratio_loss

    def _do_calc_ssl_loss_atom(self, emb, emb_side):
        emb_expanded, emb_augmented_expanded, label, shuffle_indices, repeat_indices = \
                generate_negative_sampling_cartesian_3(emb, emb, self._n_repeats)

        emb_side_expanded, emb_side_augmented_expanded, label_side, _, _ = \
            generate_negative_sampling_cartesian_3(emb_side, emb_side, self._n_repeats, shuffle_indices, repeat_indices)
        cosine_sim_side = (1.0 + calc_cosine_similarity(emb_side_expanded, emb_side_augmented_expanded)) / 2
        # batch norm
        cosine_sim_side = cosine_sim_side / (tf.reduce_max(cosine_sim_side) + 1e-6)

        similarity_measure = SimilarityMeasure(self._similarity_measure_config)
        similarity_score, _, _ = similarity_measure.calc_similarity(emb_expanded, emb_augmented_expanded)

        cosine_sim_atom = (1.0 + calc_cosine_similarity(emb_expanded, emb_augmented_expanded)) / 2
        almost_identity = tf.cast(cosine_sim_atom > self._stage1_loss_margin, dtype=tf.int32)
        almost_identity_ratio = tf.reduce_sum((1.0 - label) * tf.cast(almost_identity, tf.float32)) / tf.reduce_sum(1.0 - label)

        # 交叉熵
        almost_identity_ratio_loss = tf.compat.v1.losses.log_loss(label, similarity_score, weights=(1.0 - label) * tf.cast(almost_identity, tf.float32))

        loss = tf.compat.v1.losses.log_loss(label, similarity_score, weights=cosine_sim_side * (1.0 - (1.0 - label) * tf.cast(almost_identity, tf.float32)))
        return loss, cosine_sim_side, almost_identity_ratio, almost_identity_ratio_loss

    def do_metrics_to_show(self):
        return super().do_metrics_to_show() + ['stage1_loss', 'atom_repr_ssl_loss', 'side_repr_ssl_loss'
            , 'cosine_sim_side', 'cosine_sim_side_max', 'cosine_sim_side_min', 'cosine_sim_side_mean'
            , 'almost_identity_ratio_side', 'almost_identity_ratio_loss_side', 'almost_identity_ratio_atom', 'almost_identity_ratio_loss_atom']

    def do_metrics_to_outptut(self):
        return super().do_metrics_to_outptut() + ['atom_repr', 'side_repr', 'cosine_sim_alignment']

    def _create_mlp_and_do_mapping(self, emb, scope_prefix):
        with tf.compat.v1.variable_scope(scope_prefix):
            projection_mlp = create_mlp_layer_v2([128], self._projection_dim, None, 'projection')
            repr_projection = projection_mlp(emb)

            alignment_mlp = create_mlp_layer_v2([128], self._alignment_dim, None, 'alignment')
            emb_stop_grad = Lambda(lambda x: K.stop_gradient(x))(emb)
            repr_alignment = alignment_mlp(emb_stop_grad)

        return repr_projection, repr_alignment
