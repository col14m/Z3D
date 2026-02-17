import numpy as np
import torch
from .grounding_utils import align_points
from .mask_utils import bbox2mask


def get_instances_3d(
    scene_ann,
    point_cloud,
    axis_align_matrix
):
    """
    Load ground-truth object proposals for Nr3D benchmark.
    """
    gt_bbox_ann = scene_ann['bbox_list']
    gt_bbox_ids = []
    gt_bbox_list = []
    for item in gt_bbox_ann:
        gt_bbox_ids.append(item['id'])
        gt_bbox_list.append(item['bbox_3d'])
    aligned_pc = align_points(point_cloud, axis_align_matrix=axis_align_matrix)    
    gt_instance_masks = bbox2mask(torch.tensor(aligned_pc), torch.tensor(gt_bbox_list))
    instances_3d = dict()
    gt_bbox_lookup_table = dict()
    for i in range(len(gt_bbox_ids)):
        idx = gt_bbox_ids[i]
        bbox_mask = gt_instance_masks[:, i].flatten()
        instances_3d[idx] = np.nonzero(bbox_mask.numpy().astype('int'))   
        gt_bbox_lookup_table[idx] = gt_bbox_list[i]
    
    return instances_3d, gt_bbox_ids, gt_bbox_list, gt_bbox_lookup_table 