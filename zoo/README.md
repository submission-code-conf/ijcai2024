###Code structure

Modules such as self-supervised data augmentation, self-supervised comparison, and endogenous structures are abstracted into sub_task. All sub_tasks inherit sub_task_base, and sub_task is similar to keras layer.
, the difference is that sub_task_base explicitly defines the loss interface and output, the benefit of sub_task is that each task can be tested and evaluated independently

All sub_tasks in a pipeline form a directed acyclic graph. The output of the upstream sub_task can be the input of the downstream sub_task. The entire pipeline is assembled by multi_task_assembler. A configuration example is given below.

```python
{
sub_task_names = ["query_ssl_augment", "item_ssl_augment", "ssl", "ltr"]
,ltr = { name = "ltr"
    , sub_task_class_name = "PairwiseLTRSubTask"
    , hidden_units_list = [128, 32]
    , is_exclusive = False
    , add_prefix = true
    , loss_weight = 0.0
    
    , rank_label_col = "score_level"
    , rank_score_col = "ssl_qi_score"
    , session_id_col = "task_id_hash"
    , skip_calc_loss = True
    }
    
,ssl = { name = "ssl"
    , sub_task_class_name = "SSLSubTask"
    , hidden_units_list = []
                          #                          , feature_group_list = [10, 11, 15]
    , is_exclusive = False
    , add_prefix = true
    , loss_weight = 1.0
    
    , query_embs_augmented_col = "query_ssl_augment_embs_augmented"
    , item_embs_augmented_col = "item_ssl_augment_embs_augmented"
    , in_batch_negative_sampling_repeats = 10
    , contrastive_loss_tau = 0.1
    , similarity_measure = "cosine"
    , ssl_representation_dim = 256
    , log_loss_weight = 1.0
    , ssl_loss_weight = 0.1
    , cross_ssl_loss_weight = 0.1
    , multi_head_num = 8
    }
    
,query_ssl_augment = { name = "query_ssl_augment"
    , sub_task_class_name = "SSLAugmentSubTask"
    , hidden_units_list = [128, 32]
    , feature_group_list = [10]
    
                           #                , feature_group_list = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    , is_exclusive = False
    , add_prefix = true
    , loss_weight = 0.0
    
    , atom_feature_mask_out_prob = 0.0
    , atom_feature_correlation_col = "condition_probs"
    , ssl_augment_dropout_rate = 0.1
    }
    
    
,item_ssl_augment = { name = "item_ssl_augment"
    , sub_task_class_name = "SSLAugmentSubTask"
    , hidden_units_list = [128, 32]
                          #                          , feature_group_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 23]
    , feature_group_list = [3]
    , is_exclusive = False
    , add_prefix = true
    , loss_weight = 0.0
    
    , atom_feature_mask_out_prob = 0.0
    , atom_feature_correlation_col = "condition_probs"
    , ssl_augment_dropout_rate = 0.1
    }


    , add_batch_negative_sampling = False
    , negative_sampling_strategy = "cartesian"
    , rand_shuffle_group_list = [10, 11, 12, 13, 14, 15, 20]
    , rand_shuffle_extra_input_features = ["query_semantic_emb"]
    
    , fit_loss_weight = 1.0
    
    , label_mapping = {"click_score": "is_click"}
}
```

