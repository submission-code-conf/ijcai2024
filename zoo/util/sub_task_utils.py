from sub_task.pairwise_ltr_sub_task import PairwiseLTRSubTask
from sub_task.pointwise_ltr_sub_task import PointwiseLTRSubTask
from sub_task.simple_decomposition_similarity_sub_task import SimpleDecompositionSimilaritySubTask
from sub_task.simple_decomposition_similarity_sub_task_v2 import \
    SimpleDecompositionSimilaritySubTask as SimpleDecompositionSimilaritySubTaskV2
from sub_task.two_tower_sub_task import TwoTowerSubTask
from sub_task.endogeny_decomposition_sub_task import EndogenyDecompositionSubTask
from sub_task.endogeny_decomposition_sub_task_v2 import \
    EndogenyDecompositionSubTask as EndogenyDecompositionSubTaskV2
from sub_task.glm_sub_task import GLMSubTask
from sub_task.mlp_sub_task import MLPSubTask
from sub_task.multi_class_sub_task import MultiClassSubTask
from sub_task.ssl_augment_sub_task import SSLAugmentSubTask
from sub_task.ssl_sub_task import SSLSubTask


def get_sub_task_class(kls):
    model_cache = {
        'MLPSubTask': MLPSubTask
        , 'SSLAugmentSubTask': SSLAugmentSubTask
        , 'SSLSubTask': SSLSubTask
        , 'MultiClassSubTask': MultiClassSubTask
        , 'GLMSubTask': GLMSubTask
        , 'PairwiseLTRSubTask': PairwiseLTRSubTask
        , 'EndogenyDecompositionSubTask': EndogenyDecompositionSubTask
        , 'SimpleDecompositionSimilaritySubTask': SimpleDecompositionSimilaritySubTask
        , 'EndogenyDecompositionSubTaskV2': EndogenyDecompositionSubTaskV2
        , 'SimpleDecompositionSimilaritySubTaskV2': SimpleDecompositionSimilaritySubTaskV2
        , 'PointwiseLTRSubTask': PointwiseLTRSubTask
        , 'TwoTowerSubTask': TwoTowerSubTask
                   }
    return model_cache[kls]