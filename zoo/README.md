###代码结构

自监督数据增广，自监督对比，内生结构等模块抽象成为sub_task, 所有sub_task继承sub_task_base, sub_task类似keras layer
，区别在于sub_task_base显式定义了loss接口和输出，sub_task好处在每个task可以单独测试和评估

一个pipeline中所有sub_task构成一个有向无环图， 上游sub_task的输出可以下游sub_task的输入，整个pipeline由multi_task_assembler负责装配， 下面给出一个配置示例

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

