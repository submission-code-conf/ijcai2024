class ZooConstants:
    # AMOUNT_RANGE_UB = "None"
    COSINE_SCALE_TAU = 'cosine_scale_tau'
    N_REPEATS = "n_repeats"
    SIMILARITY_MEASURE_CONFIG = "similarity_measure_config"
    LOG_LOSS_WEIGHT = "log_loss_weight"
    MULTI_HEAD_NUM = "multi_head_num"
    LOGIT_COL = "logit_col"
    SAMPLE_WEIGHT_COL = "sample_weight_col"
    MULTI_CLASS_LABEL_CLASSES_SAMPLED = "multi_class_label_classes_sampled"
    MULTI_CLASS_LABEL_CLASSES = "multi_class_label_classes"
    MULTI_CLASS_LABEL_COL = "multi_class_label_col"
    CROSS_SSL_LOSS_WEIGHT = "cross_ssl_loss_weight"
    SKIP_CALC_LOSS = "skip_calc_loss"
    SESSION_ID_COL = "session_id_col"
    RANK_LABEL_COL = "rank_label_col"
    RANK_SCORE_COL = "rank_score_col"
    SSL_REPRESENTATION_DIM = "ssl_representation_dim"
    SIMILARITY_MEASURE = "similarity_measure"
    SIMILARITY_KERNEL_SIZE = "similarity_kernel_size"
    SKIP_VALIDATION = "skip_validation"
    ITEM_EMBS_AUGMENTED_COL = "item_embs_augmented_col"
    QUERY_EMBS_AUGMENTED_COL = "query_embs_augmented_col"
    SSL_LOSS_WEIGHT = "ssl_loss_weight"
    ATOM_FEATURE_CORRELATION_COL = "atom_feature_correlation_col"
    SSL_AUGMENT_DROPOUT_RATE = "ssl_augment_dropout_rate"
    ATOM_FEATURE_MASK_OUT_PROB = "atom_feature_mask_out_prob"

    FIELDS_MASK = "fields_mask"
    CONTRASTIVE_LOSS_TAU = "contrastive_loss_tau"
    INPUT_BATCH_SIZE = "input_batch_size"
    IN_BATCH_NEGATIVE_SAMPLING_REPEATS = "in_batch_negative_sampling_repeats"
    MASK_COLS = "mask_cols"
    INPUTS_FROM_TASK_QUERY = "inputs_from_task_query"
    INPUTS_FROM_EXTRA_QUERY = "inputs_from_extra_query"
    INPUTS_FROM_TASK_ITEM = "inputs_from_task_item"
    INPUTS_FROM_EXTRA_ITEM = "inputs_from_extra_item"


    DEPENDENCY_SUB_TASKS = "dependency_sub_tasks"
    TREATMENT_MONOTONE_DECREASING_DICT = "treatment_monotone_decreasing_dict"
    TREATMENT_COLS = "treatment_cols"
    TREATMENT_BUCKETS_DICT = 'treatment_bucket_dict'
    LABEL_COL = 'label_col'
    TREATMENT = "treatment"
    INPUTS_FROM_TASK = "inputs_from_task"
    INPUTS_FROM_EXTRA = "inputs_from_extra"
    QUERY_INTENT_LEVEL_CNT_COL = "query_intent_level_cnt_col"
    ITEM_INTENT_CATEGORY_ID = "item_intent_category_id"
    CLASSES = "classes"
    NEGATIVE_SAMPLING_STRATEGY = "negative_sampling_strategy"
    STRUCTURAL_FIELDS_START_INDEX = "structural_fields_start_index"
    RAND_SHUFFLE_EXTRA_INPUT_FEATURES = "rand_shuffle_extra_input_features"
    SEMANTIC_EMB_SIZE = "semantic_emb_size"
    QUERY_SEMANTIC_EMB = "query_semantic_emb"
    ITEM_SEMANTIC_EMB = "item_semantic_emb"
    MASK_COL = "mask_col"
    IN_BATCH_SAMPLING_MASK = "in_batch_sampling_mask"
    RAND_SHUFFLE_GROUP_LIST = "rand_shuffle_group_list"
    ADD_BATCH_NEGATIVE_SAMPLING = "add_batch_negative_sampling"
    COSINE_EMBEDDING_LOSS_MARGIN = 'cosine_embedding_loss_margin'
    FIELD_SIMILARITY_LABEL = "field_similarity_label"
    TREATMENT_COL = "treatment_col"
    ADD_PREFIX = "add_prefix"
    HIDDEN_UNITS_LIST = "hidden_units_list"
    IS_EXCLUSIVE = "is_exclusive"
    FEATURE_GROUP_LIST = "feature_group_list"
    SVM_GROUP_LIST = "svm_group_list"
    LOSS_WEIGHT = "loss_weight"
    IS_COMPONENT = False
    DEPENDENCY_SCORE_KEY_MAPPING = "dependency_score_key_mapping"
    QUERY_TYPE_GROUP_LIST = "query_type_group_list"
    SUB_TASK_CLASS_NAME = "sub_task_class_name"
    SUB_TASK_NAMES = "sub_task_names"
    SUB_TASK_NAMES_DEFAULT = []
    QUERY_TYPE_EMBEDDING = "query_type_embedding"
    QUERY_STRUCTURAL_EMB_LIST = "query_structural_em_list"
    ITEM_STRUCTURAL_EMB_LIST = "item_structural_emb_list"

    SIGMA = "sigma"
    MU = "u"
    CONFIG_NAMES_UNCERTAINTY_DEFAULT = []
    CONFIG_NAMES_UNCERTAINTY = "config_names_uncertainty"
    GAUSSIAN_NLL_LOSS_EPS_DEFAULT = 1e-6
    GAUSSIAN_NLL_LOSS_EPS = "gaussian_nll_loss_eps"
    TOTAL_LOSS = "total_loss"
    WEIGHT_LABEL_KEY_DICT_DEFAULT = {}
    WEIGHT_LABEL_KEY_DICT = "weight_label_key_dict"


    AMOUNT_RANGE_UB_DEFAULT = -1
    AMOUNT_RANGE_UB = "amount_range_ub"
    MONOTONIC_DECREASING_DEFAULT = False
    MONOTONIC_DECREASING = "monotonic_decreasing"

    REG_WEIGHT = "reg_weight"
    QUERY_STRUCTURAL_GROUP_LIST = "query_structural_group_list"
    ITEM_STRUCTURAL_GROUP_LIST = "item_structural_group_list"
    DSMN_HPARAMS = "dsmn_hparams"
    ISOTONIC_LOSS_WEIGHT = "isotonic_loss_weight"
    ISOTONIC_LOSS_WEIGHT_DEFAULT = 0.0
    ISOTONIC_LOSS_WEIGHT_INITIAL = "isotonic_loss_weight_initial"
    ISOTONIC_LOSS_WEIGHT_STEP_DECAY = "isotonic_loss_weight_step_decay"
    ISOTONIC_LOSS_WEIGHT_STEP_DECAY_DEFAULT = 0.0

    R_DROP_MODE = "r_drop_mode"
    R_DROP_LOSS_WEIGHT = "r_drop_loss_weight"
    PREDICT_SCORE_KEY = "predict_score_key"
    R_DROP_LOSS = "r_drop_loss"

    PHASE_UW = "phase_uw"
    PHASE_BIAS = "phase_bias"

    PHASE = "phase"
    PHASE_DEFAULT = "phase_both"

    EXTRA_INPUTS = "extra_inputs"
    EXTRA_INPUTS_DEFAULT = []

    UW_CENTRALIZATION_LOSS_WEIGHT = "uw_centralization_loss_weight"
    UW_CENTRALIZATION_LOSS_WEIGHT_DEFAULT = 0.0

    SAMPLE_WEIGHT_USING = "sample_weight_using"
    SAMPLE_WEIGHT_USING_DEFAULT = {}

    AMOUNT_STEP_LIMIT = "amount_step_limit"
    AMOUNT_STEP_LIMIT_DEFAULT = 100
    AMOUNT_STEP_SIZE = "amount_step_size"
    AMOUNT_STEP_SIZE_DEFAULT = 1.0

    DO_SIGMOID_TRANSFORM = "do_sigmoid_transform"
    DO_SIGMOID_TRANSFORM_DEFAULT = 1

    CALIBRATE = "calibrate"
    DO_CALIBRATE_TRAIN = "do_calibrate_train"
    DO_CALIBRATE_TRAIN_DEFAULT = 0
    CALIBRATE_LOSS_WEIGHT_DEFAULT = 0.0
    CALIBRATE_LOSS_WEIGHT = "calibrate_loss_weight"

    TASK_WEIGHT_DICT_DEFAULT = {}
    TASK_WEIGHT_DICT = "task_weight_dict"

    NEGATIVE_SAMPLING_RATIO_DEFAULT = 1.0
    NEGATIVE_SAMPLING_RATIO = "negative_sampling_ratio"
    NEGATIVE_SAMPLING_RATIO_MAPPING_DEFAULT = {}
    NEGATIVE_SAMPLING_RATIO_MAPPING = "negative_sampling_ratio_mapping"

    UPLIFT_WEIGHT_PRIOR_DEFAULT = 0.0
    UPLIFT_WEIGHT_PRIOR = "uplift_weight_prior"

    BEHAVIOUR_SEQ_CONFIG_DEFAULT = []
    BEHAVIOUR_SEQ_CONFIG = "behaviour_seq_config"

    DCN_GROUP_LIST = "dcn_group_list"
    DCN_ORDER = "dcn_order"
    DCN_ORDER_DEFAULT = 1

    FEATURE_GROUP_BLACKLIST_FITLER_FEATURE = "feature_group_blacklist_fitler_feature"

    FEATURE_GROUP_BLACKLIST = "feature_group_blacklist"
    FEATURE_GROUP_BLACKLIST_DEFAULT = []

    FEATURE_GROUP_WHITELIST = "feature_group_whitelist"
    FEATURE_GROUP_WHITELIST_DEFAULT = []

    USE_DENSE_INPUT_DEFAULT = 0
    USE_DENSE_INPUT = "use_dense_input"

    USE_SPARSE_INPUT_DEFAULT = 1
    USE_SPARSE_INPUT = "use_sparse_input"

    AMOUNT_TRUNCATE_UB_DEFAULT = 1e6
    AMOUNT_TRUNCATE_UB = "amount_truncate_ub"
    ISOTONIC_LOSS_EPSILON_DEFAULT = 0.0
    ISOTONIC_LOSS_EPSILON = "isotonic_loss_epsilon"

    EXPAND_SIZE = "expand_size"
    EXPAND_SIZE_DEFAULT = 1

    SAMPLE_WEIGHT = "sample_weight"
    # USE_SAMPLE_WEIGHT = 'use_sample_weight'
    # USE_SAMPLE_WEIGHT_DEFAULT = 0
    SAMPLE_WEIGHT_FIELD = "sample_weight"
    SAMPLE_WEIGHT_FIELD_DEFAULT = None

    MIC_THRES = "mic_thres"
    MIC_THRES_DEFAULT = 10000

    NOISE_LOSS_WEIGHT_DEFAULT = 1e-6
    NOISE_LOSS_WEIGHT = "noise_loss_weight"

    BEST_AUC_FILTER_SCORE_NAMES = "best_auc_filter_score_names"
    BEST_LOGLOSS_FILTER_SCORE_NAMES = "best_logloss_filter_score_names"

    N_ATTENTIONS_ISOTONIC_LAYER = "n_attentions_isotonic_layer"
    N_ATTENTIONS_ISOTONIC_LAYER_DEFAULT = 1

    AMOUNT_BUCKETS = 'amount_buckets_conf'

    EPR_THRES_DEFAULT = 0.8
    EPR_THRES = "epr_thres"

    LABEL_KEY_DEPENDENCIES = "label_key_dependencies"
    LABEL_KEY_DEPENDENCIES_DEFAULT = []

    DIPN_CONFIG_NAMES = "dipn_config_names"
    DIPN_CONFIG_NAMES_DEFAULT = ['config_dipn']

    CONTROL_FEATURE_UB = "CONTROL_FEATURE_UB"
    CONTROL_FEATURE_UB_DEFAULT = 10

    USER_PROFILE_COL = "user_profile"
    USER_PROFILE_DENSE = "user_profile_dense"
    SIGN_SCOPE = 'kpi_score'
    WO_SCOPE = 'use_score'
    USER_PROFILE_EMBEDDING = 'user_profile_embedding'
    USER_PROFILE_EMBEDDING_UPLIFT = 'user_profile_embedding_uplift'
    AMOUNT_COL = 'amount'
    MAX_AMOUNT_COL = 'max_amount'
    LABEL = 'label'
    SCORE = 'score'
    CONTROL_FEATURE = 'control_feature'

    ISOTONIC_LOSS = 'isotonic_loss'

    UPLIFT = "uplift"

    SPARSE_EMBEDDING_DIM = 'sparse_embedding_dim'
    SPARSE_EMBEDDING_DIM_DEFAULT = 32

    MODEL_CONFIG = "model_config"

    DATASET = 'dataset'
    SWITCH_TO_AISTUDIO = False

    ISOTONIC_LOSS_CLASS = "isotonic_loss_class"
    ISOTONIC_LOSS_CLASS_DEFAULT = "IsotonicLossFirstOrder"
