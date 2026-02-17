import numpy as np
from sklearn.neighbors import KDTree
import torch
from torch_scatter import scatter_mean


def remap_segmentation(source_points: np.ndarray, 
                       source_labels: np.ndarray,
                       target_points: np.ndarray) -> np.ndarray:
    """
    Remaps segmentation labels from source_points to target_points using 1-NN.

    Parameters:
    - source_points: (N, 3) array of XYZ coordinates of source point cloud
    - source_labels: (N,) array of integer labels corresponding to source_points
    - target_points: (M, 3) array of XYZ coordinates of target point cloud

    Returns:
    - target_labels: (M,) array of labels mapped from source_points
    """
    kdtree = KDTree(source_points)
    _, idx = kdtree.query(target_points, k=1)
    target_labels = source_labels[idx.flatten()]
    return target_labels
    

def trim_mask_by_superpoints(inside_bbox: torch.Tensor, 
                             sp_pts_mask: torch.Tensor,  
                             point: torch.Tensor, 
                             return_bin_mask: bool = False,
                             low_sp_thr: int = 0.18, 
                             up_sp_thr: int = 0.81) -> torch.Tensor:
        
        """Trim binary point mask with corresponding superpoint mask.

        Args:
            inside_bbox (Tensor): Boolean mask indicating whether each point
                lies inside the bounding box.
            sp_pts_mask (Tensor): Superpoint assignment for each point.
            point (Tensor): Point cloud coordinates.
            return_bin_mask (bool): Whether to additionally return the trimmed
                binary point mask. Defaults to False.
            low_sp_thr (float): Lower threshold for removing superpoints.
                Defaults to 0.18.
            up_sp_thr (float): Upper threshold for adding superpoints.
                Defaults to 0.81.

        Returns:
            Tensor or tuple:
                - Tensor: Points selected by the trimmed mask.
                - (optional) Tensor: Trimmed binary mask over points if
                ``return_bin_mask`` is True.
        """
        inside_bbox = inside_bbox.unsqueeze(0)
        point = point.unsqueeze(1)
        sp_inside = scatter_mean(inside_bbox.float(),
                                 sp_pts_mask, dim=-1)
        sp_del = sp_inside < low_sp_thr
        inside_bbox[sp_del[:, sp_pts_mask]] = False

        sp_add = sp_inside > up_sp_thr
        inside_bbox[sp_add[:, sp_pts_mask]] = True

        if return_bin_mask:
            return point[inside_bbox.T.bool()], inside_bbox.squeeze(0)
        else:
            return point[inside_bbox.T.bool()]


def bbox2mask(points: torch.Tensor, 
              bboxes: torch.Tensor) -> torch.Tensor:
    """
    Compute a boolean mask indicating which points are inside which bounding boxes.

    Args:
        points (Tensor): Point cloud of shape (N, 3)
        bboxes (Tensor): Bounding boxes of shape (M, 6) 
                        (center_x, center_y, center_z, dx, dy, dz)

    Returns:
        Tensor: Boolean mask of shape (N, M).
        mask[i, j] == True if point i is inside bbox j.
    """
    N = points.shape[0]
    M = bboxes.shape[0]
    points_exp = points.unsqueeze(1).expand(N, M, 3) 
    bboxes_exp = bboxes.unsqueeze(0).expand(N, M, 6)

    bbox_min = bboxes_exp[..., :3] - bboxes_exp[..., 3:] / 2 
    bbox_max = bboxes_exp[..., :3] + bboxes_exp[..., 3:] / 2

    inside = (points_exp >= bbox_min) & (points_exp <= bbox_max) 
    mask = inside.all(dim=-1)                                     

    return mask


def project_masks2pcd(mask_3d_list, point_cloud):
    N = point_cloud.shape[0]
    final_bin_masks = []
    for mask_3d in mask_3d_list:
        mask_3d = mask_3d[:, :3]
        if mask_3d.shape[0] == 0:
            continue
        old_labels = np.arange(N)
        new_labels = remap_segmentation(point_cloud, old_labels, mask_3d)
        new_labels = np.unique(new_labels)
        binary_mask = np.zeros(N)
        binary_mask[new_labels] = 1
        final_bin_masks.append(binary_mask)
    return final_bin_masks