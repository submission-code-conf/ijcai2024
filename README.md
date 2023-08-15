# Causal Embedding with Dual Feature Decomposition for Addressing Exposure Bias in Recommender Systems

## Overview
Recommender systems often grapple with the challenge of exposure bias. This phenomenon arises when users predominantly interact with a narrow set of items (head items), leading to the misconception that non-exposed items (tail items) are of no interest to them. Existing solutions, such as inverse propensity scoring and causal embeddings, sometimes fail to delve deep into the latent relationships between users, items, and interactions, a challenge known as the endogeneity problem.

Our research introduces the **Dual Feature Decomposition (DFD)** approach to tackle this issue. By decomposing features into atomic and side features, we aim to provide more accurate and unbiased recommendations.

## Key Features
- **Dual Feature Decomposition**: Breaks down features into atomic (related to bias and unobserved factors) and side (observable without bias) features.
- **Neural Network Mappings**: Designed to minimize the influence of the atomic space on side representations.
- **Vertical Projection**: Applied to further reduce bias in recommendations.

## Background
The endogeneity problem suggests that unseen factors might influence a user's preference. For instance, a user might prefer a movie due to a specific actor, even if they haven't interacted with it. Our model aims to capture such nuances to provide a more holistic recommendation.

## Related Work
We delve into related research, focusing on advancements in causal embeddings and representation disentanglement, to provide context and highlight the novelty of our approach.

## Code Structure and Usage
Self-supervised data augmentation, self-supervised contrast, and endogenous structure modules are abstracted as `sub_task`. All `sub_task` inherit from `sub_task_base`. The `sub_task` is similar to a Keras layer, but the difference is that `sub_task_base` explicitly defines the loss interface and output. The advantage of `sub_task` is that each task can be tested and evaluated independently.

All `sub_task` in a pipeline form a directed acyclic graph. The output of the upstream `sub_task` can be the input of the downstream `sub_task`. The entire pipeline is assembled by `multi_task_assembler`. Below is a configuration example:

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

