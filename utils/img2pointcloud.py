import numpy as np
import os
from PIL import Image
import math
import cv2
import torch
from typing import List, Tuple, Dict

POPULAR_COLORS = [
    np.array([255, 0, 0]),
    np.array([0, 255, 0]),
    np.array([255, 51, 255]),
    np.array([0, 0, 255]),
    np.array([255, 255, 51]),
    np.array([153, 51, 255]),
    np.array([255, 153, 51]),
    np.array([0, 255, 255]),
    np.array([255, 0, 127]),
]


def get_adapted_intrinsic(intrinsic: np.ndarray, 
                          desired_resolution: Tuple[int, int], 
                          original_resolution: Tuple[int, int]) -> np.ndarray:
    """Adjust camera intrinsic matrix to match a target resolution.

    Args:
        intrinsic (ndarray): Original 3x3 camera intrinsic matrix.
        desired_resolution (tuple[int, int]): Target resolution.
        original_resolution (tuple[int, int]): Original resolution.

    Returns:
        ndarray: 3x3 adapted intrinsic matrix adapted for the desired resolution.
    """
    
    if original_resolution == desired_resolution:
        return intrinsic
    
    resize_width = int(math.floor(desired_resolution[1] * float(
                    original_resolution[0]) / float(original_resolution[1])))
    
    adapted_intrinsic = intrinsic.copy()
    adapted_intrinsic[0, 0] *= float(resize_width) / float(original_resolution[0])
    adapted_intrinsic[1, 1] *= float(desired_resolution[1]) / float(original_resolution[1])
    adapted_intrinsic[0, 2] *= float(desired_resolution[0] - 1) / float(original_resolution[0] - 1)
    adapted_intrinsic[1, 2] *= float(desired_resolution[1] - 1) / float(original_resolution[1] - 1)
    return adapted_intrinsic


def _write_obj(points: np.ndarray, out_filename: str):
    """Write points into `obj` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def load_scan(pcd_path: str) -> np.ndarray:
    """Load a point cloud from a binary file.

    Args:
        pcd_path (str): Path to the binary point cloud file (.bin).

    Returns:
        ndarray: Nx6 array with point cloud data.
    """
    pcd_data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 6)
    return pcd_data


def img2pointcloud(root_dir: str, 
                   scene_id: str, 
                   img_idx: int, 
                   desired_resolution: Tuple[int, int], 
                   n_points: int = 10000, 
                   masks: List[np.ndarray] = None, 
                   write: bool = True, 
                   return_R_K: bool = False, 
                   data_prefix: Dict[str, str] = None) -> tuple[np.ndarray, List[np.ndarray]]:
    
    """Project single RGB-D image to 3D space.

    Args:
        root_dir (str): Root directory containing scene data.
        scene_id (str): Scene identifier.
        img_idx (int): Index of the image/frame in the scene.
        desired_resolution (tuple[int, int]): Target image resolution (height, width).
        n_points (int, optional): Number of points to sample from the full
            point cloud. Defaults to 10000.
        masks (list[np.ndarray], optional): List of 2D masks to apply to
            the point cloud. Defaults to None.
        write (bool, optional): Whether to save the generated point clouds
            as OBJ files. Defaults to True.
        return_R_K (bool, optional): Whether to return the adapted intrinsic
            matrix and extrinsic matrix. Defaults to False.
        data_prefix (dict, optional): Dictionary specifying data subfolders. 
            Defaults to None.

    Returns:
        tuple:
            - ndarray: Nx6 array of sampled colored point cloud (XYZRGB).
            - list[ndarray]: List of masked point clouds, one per mask.
            - (optional) ndarray: Adapted intrinsic matrix if `return_R_K` is True.
            - (optional) ndarray: Extrinsic matrix if `return_R_K` is True.
    """
    if data_prefix is None:
        data_prefix = {}
    
    if write:
        original_pointcloud_path = os.path.join(root_dir, 
                                            data_prefix.get('points', 'points'), 
                                            f'{scene_id}.bin')
        original_pointcloud = load_scan(original_pointcloud_path)
        _write_obj(original_pointcloud, 'original_pointcloud.obj')

    posed_image_prefix = data_prefix.get('posed_images', 'posed_images')
    intrinsic_path = os.path.join(root_dir, 
                                  posed_image_prefix,
                                  scene_id, 
                                  'intrinsic.txt')
    frame_path = os.path.join(root_dir, 
                              posed_image_prefix, 
                              scene_id, str(img_idx).zfill(5) + '.jpg')
    extrinsic_path = frame_path.replace('.jpg', '.txt')
    depth_path = frame_path.replace('.jpg', '.png')

    frame = Image.open(frame_path).convert('RGB')
    original_resolution = (frame.size[1], frame.size[0])
    frame = frame.resize((desired_resolution[1], desired_resolution[0]))

    new_masks = []
    if masks:
        for mask in masks:
            mask = cv2.resize(mask.astype('float32'), 
                              (desired_resolution[1], desired_resolution[0]), 
                              interpolation=cv2.INTER_NEAREST)
            new_masks.append(mask.astype('int').astype('bool'))

    extrinsic = np.loadtxt(extrinsic_path)  
    depth = Image.open(depth_path)
    depth = np.array(depth, dtype=np.float32) / 1000.0  # Convert to meters
    assert depth.shape == desired_resolution
    intrinsic = np.loadtxt(intrinsic_path)

    # Create pixel coordinates
    height, width = depth.shape
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    # Flatten coordinates and depth
    pixels = np.column_stack((xx.flatten(), yy.flatten(), 
                            np.ones_like(xx.flatten())))
    depth_flat = depth.flatten()
    # Get colors from the frame
    frame_array = np.array(frame)
    colors = frame_array.reshape(-1, 3)
    # Filter out invalid depth values
    valid = depth_flat > 0
    pixels = pixels[valid]
    depth_flat = depth_flat[valid]
    colors = colors[valid]
    if new_masks:
        for i in range(len(new_masks)):
            new_masks[i] = new_masks[i].flatten()[valid]

    intrinsic_adapted = get_adapted_intrinsic(intrinsic, 
                                            desired_resolution, 
                                            original_resolution)
    # Convert to camera coordinates
    camera_coords = (np.linalg.inv(intrinsic_adapted[:3, :3]) 
                    @ pixels.T).T * depth_flat[:, np.newaxis]
    # Convert to world coordinates
    world_coords = (extrinsic @ np.column_stack((camera_coords, 
                                                np.ones_like(camera_coords[:,
                                                            0]))).T).T[:, :3]
    # Combine world coordinates with colors
    colored_point_cloud = np.hstack((world_coords, colors))
    N_pcd = colored_point_cloud.shape[0]
    colored_point_cloud_sp_idx = np.random.choice(np.arange(N_pcd), 
                                                  size=min(n_points, N_pcd), 
                                                  replace=False)
    colored_point_cloud = colored_point_cloud[colored_point_cloud_sp_idx]
    if write:
        _write_obj(colored_point_cloud, f'img_{img_idx}_pointcloud.obj')
    mask_res = []
    if new_masks:
        for i in range(len(new_masks)):
            mask = new_masks[i][colored_point_cloud_sp_idx]
            mask_pc = colored_point_cloud[mask]
            new_color = POPULAR_COLORS[i % len(POPULAR_COLORS)]
            mask_pc[:, 3:] = new_color
            mask_res.append(mask_pc)
            if write:
                _write_obj(mask_pc, f'mask_{i}_pc.obj')
    if return_R_K:
        return colored_point_cloud, mask_res, intrinsic_adapted, extrinsic
    return colored_point_cloud, mask_res


def multiple_img2pointcloud(root_dir: str, 
                            scene_id: str, 
                            img_indices: List[int], 
                            desired_resolution: Tuple[int, int], 
                            n_points: int = 10000, 
                            final_n_points: int = 50000, 
                            masks: List[List[np.ndarray]] = None, 
                            write: bool = True, 
                            data_prefix: Dict[str, str] = None) -> tuple[np.ndarray, List[np.ndarray]]:
    """Project multiple RGB-D images to 3D space.
       Return the aggregated point cloud and projected masks. 

    Args:
        root_dir (str): Root directory containing scene data.
        scene_id (str): Scene identifier.
        img_indices (list[int]): List of image/frame indices to process.
        desired_resolution (tuple[int, int]): Target image resolution (height, width).
        n_points (int, optional): Number of points to sample from per-image point clouds.
            Defaults to 10000.
        final_n_points (int, optional): Maximum number of points in final point cloud.
            Defaults to 50000.
        masks (list[list[np.ndarray]], optional): List of per-image masks.
            Defaults to None.
        write (bool, optional): Whether to save the point clouds and masks as OBJ files.
            Defaults to True.
        data_prefix (dict, optional): Dictionary specifying data subfolders. 
            Defaults to None.

    Returns:
        tuple:
            - ndarray: Nx6 array of aggregated point cloud (XYZRGB).
            - list[ndarray]: List of masked point clouds, one per mask.
    """
    final_pc = np.array([])
    final_masks = [np.array([]) for _ in range(len(masks[0]))]

    for i, img_idx in enumerate(img_indices):
        pc_i, masks_i = img2pointcloud(root_dir, 
                                       scene_id, img_idx, 
                                       desired_resolution=desired_resolution, 
                                       n_points=n_points, 
                                       masks=masks[i], 
                                       write=False, 
                                       data_prefix=data_prefix)
        if i != 0:
            final_pc = np.vstack((final_pc, pc_i))
            for j in range(len(final_masks)):
                final_masks[j] = np.vstack((final_masks[j], masks_i[j]))
        else:
            final_pc = pc_i
            for j in range(len(final_masks)):
                final_masks[j] = masks_i[j]
    N_pcd_final = final_pc.shape[0]
    final_pc_idxs_sp = np.random.choice(np.arange(N_pcd_final), 
                                        size=min(final_n_points, N_pcd_final), 
                                        replace=False)
    final_pc = final_pc[final_pc_idxs_sp]
    mask_res = []
    if write:
        pcd_fname = f'final_pc_{img_indices[0]}-{img_indices[-1]}.obj'
        _write_obj(final_pc, pcd_fname)
    for k, mask_k in enumerate(final_masks):
        N_mask = mask_k.shape[0]
        final_mask_idxs_sp = np.random.choice(np.arange(N_mask), 
                                              size=min(final_n_points, N_mask), 
                                              replace=False)
        mask_k = mask_k[final_mask_idxs_sp]
        mask_res.append(mask_k)
        if write:
            mask_fname = f'final_mask_{k}_{img_indices[0]}-{img_indices[-1]}.obj'
            _write_obj(mask_k, mask_fname)

    return final_pc, mask_res