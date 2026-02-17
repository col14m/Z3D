import numpy as np
from collections import Counter


def iou_1d(mask1, mask2) -> float:
    """Compute the Intersection over Union (IoU) for 1D binary masks.

    Args:
        mask1 (array-like): 1D array.
        mask2 (array-like): 1D array.

    Returns:
        float: IoU score between 0.0 and 1.0.
    """

    m1 = np.array(mask1).astype(bool)
    m2 = np.array(mask2).astype(bool)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def iou_matching(pred_bin_masks: list[np.ndarray], 
                 gt_proposals: dict[int, np.ndarray], 
                 point_cloud: np.ndarray) -> list[int | None]:
    """Match predicted binary masks to ground-truth proposals using 1D IoU.

    Args:
        pred_bin_masks (list[np.ndarray]): List of predicted 1D binary masks.
        gt_proposals (dict[int, array-like]): Dictionary mapping proposal IDs
            to indices of points belonging to each ground-truth mask.
        point_cloud (np.ndarray): Point cloud.

    Returns:
        list[int | None]: List of matched ground-truth proposal IDs, one per
            predicted mask. If no proposal achieves a positive IoU, the
            corresponding entry will be None.
    """
    N = point_cloud.shape[0]
    matched_mask_ids = []
    for bin_mask in pred_bin_masks:
        max_iou = 0
        best_mask = None
        for mask_id in gt_proposals.keys():
            ids = gt_proposals[mask_id]
            curr_binary_mask = np.zeros(N)
            curr_binary_mask[ids] = 1
            iou = iou_1d(bin_mask, curr_binary_mask)
            if iou > max_iou:
                max_iou = iou
                best_mask = mask_id
        matched_mask_ids.append(best_mask)
    
    return matched_mask_ids
    
    
def majority_voting(matched_mask_ids: list[int]) -> int:
    """Select a mask ID by majority voting.

    Args:
        matched_mask_ids (list[int]): List of matched mask IDs.

    Returns:
        int | None: The selected mask ID after majority voting. If `None`
            is the most frequent value, `None` may be returned.
    """
    counts = Counter(matched_mask_ids)
    max_count = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_count]
    # choose the one that appears first
    best_mask_id = min(candidates, key=lambda k: matched_mask_ids.index(k))
    return best_mask_id


def mv_aggregation(pred_bin_masks: list[np.ndarray], 
                   gt_proposals: dict[int, np.ndarray], 
                   point_cloud: np.ndarray) -> int | None:
    """Aggregate mask predictions from multiple views.

    This function first matches each predicted binary mask to a ground-truth
    proposal using IoU, then removes unmatched predictions, and finally
    selects the most consistent ground-truth mask ID via majority voting.

    Args:
        pred_bin_masks (list[np.ndarray]): List of predicted 1D binary masks.
        gt_proposals (dict[int, array-like]): Dictionary mapping proposal IDs
            to indices of points belonging to each ground-truth mask.
        point_cloud (np.ndarray): Point cloud.

    Returns:
        int | None: The selected ground-truth mask ID after aggregation.
            Returns None if no predicted masks could be matched.
    """
    
    matched_mask_ids = iou_matching(pred_bin_masks, gt_proposals, point_cloud)
    matched_mask_ids = [i for i in matched_mask_ids if i != None]
    if len(matched_mask_ids) == 0:
        return None
    best_mask_id = majority_voting(matched_mask_ids)
    return best_mask_id