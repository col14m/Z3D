import os
import numpy as np
from collections import defaultdict
from .mask_utils import remap_segmentation
from .grounding_utils import align_points
import torch

def get_mask_matching_mc(scene_id, args):
    '''
    Load preprocessed object proposals from MaskClustering
    '''
    object_dict = np.load(os.path.join(args['olt_root_path'], scene_id, 'object_dict.npy'), allow_pickle=True).item()
    frame_2d_mask_to_3d = defaultdict(lambda: defaultdict(int))
    masks_3d = dict()
    for mask_id_3d, data in object_dict.items():
        for mask_info in data['mask_list']:
            frame_id = mask_info[0]
            mask_id_2d = mask_info[1]
            frame_2d_mask_to_3d[frame_id][mask_id_2d] = mask_id_3d + 1
        masks_3d[mask_id_3d + 1] = data['point_ids']
    return frame_2d_mask_to_3d, masks_3d


def get_mask3d_preds(scene_id, args):
    '''
    Load raw object proposals from Mask3D
    '''
    mask_data = np.load(os.path.join(args['olt_root_path'], f'{scene_id}.npz'), allow_pickle=True)
    mask_dict = {key: mask_data[key] for key in mask_data.files}
    masks = mask_dict['ins_pcds']
    scores = mask_dict['ins_scores']
    masks_3d = dict()
    for mask_id_3d in range(len(masks)):
        curr_mask = masks[mask_id_3d][:, :3]
        masks_3d[mask_id_3d + 1] = curr_mask
    return masks_3d, scores


def get_mask_matching_mask3d(scene_id, args):
    '''
    Load preprocessed object proposals from Mask3D
    '''
    masks_3d_preds, scores = get_mask3d_preds(scene_id, args)
    masks_3d = dict()
    point_cloud = args['point_cloud']
    N = point_cloud.shape[0]
    old_labels = np.arange(N)
    for key_i, key in enumerate(masks_3d_preds.keys()):
        if float(scores[key_i]) < args['score_thr']:
            continue
        mask_3d_pcd = masks_3d_preds[key]
        if mask_3d_pcd.shape[0] == 0:
            continue
        inv_transform = np.linalg.inv(args['axis_align_matrix'])
        mask_3d_pcd = align_points(mask_3d_pcd, inv_transform)
        new_labels = remap_segmentation(point_cloud, old_labels, mask_3d_pcd)
        new_labels = np.unique(new_labels)
        masks_3d[key] = new_labels

    return masks_3d


def get_inst_predictions(scene_id, method, args):
    '''
    Load and return preprocessed object proposals.
    '''
    if method.lower() == 'mc':
        return get_mask_matching_mc(scene_id, args)[1]
    elif method.lower() == 'mask3d':
        return get_mask_matching_mask3d(scene_id, args)
    else:
        raise NotImplementedError('Undefined name of segmentation backbone!')
    
    
def gt_bbox_from_instances(
        object_id,
        point_cloud,
        semantic_mask,
        instance_mask,
        axis_align_matrix
):
    """
    Load ground-truth object proposals for ScanRefer benchmark.
    """
    object_mask = instance_mask == (object_id + 1)
    object_points = torch.tensor(point_cloud[object_mask])
    assert len(np.unique(semantic_mask[object_mask])) == 1
    sem_class = np.unique(semantic_mask[object_mask])
    is_multiple = sem_class in semantic_mask[~object_mask]

    object_points = align_points(object_points, axis_align_matrix)
    xyz_min = object_points.min(dim=0).values
    xyz_max = object_points.max(dim=0).values
    gt_center = (xyz_max + xyz_min) / 2
    gt_size = xyz_max - xyz_min
    gt_bbox = np.hstack((gt_center, gt_size))
    return gt_bbox, is_multiple