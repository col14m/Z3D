import os
import numpy as np
from PIL import Image
import torch
import cv2
import yaml
import json
import pickle
import sys
import contextlib
from typing import List, Optional, Tuple, Dict

class DotDict(dict):
    """Recursive dot-access dictionary."""
    def __getattr__(self, key):
        value = self.get(key)
        if isinstance(value, dict):
            value = DotDict(value)
            self[key] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class SilentAllOutput(contextlib.AbstractContextManager):
    """Context manager for suppressing all console outputs"""
    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()

        self._stdout = sys.stdout
        self._stderr = sys.stderr

        self._stdout_fd = sys.__stdout__.fileno()
        self._stderr_fd = sys.__stderr__.fileno()
        self._stdout_dup = os.dup(self._stdout_fd)
        self._stderr_dup = os.dup(self._stderr_fd)

        self._devnull = open(os.devnull, "w")

        os.dup2(self._devnull.fileno(), self._stdout_fd)
        os.dup2(self._devnull.fileno(), self._stderr_fd)

        sys.stdout = self._devnull
        sys.stderr = self._devnull

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self._stdout_dup, self._stdout_fd)
        os.dup2(self._stderr_dup, self._stderr_fd)

        self._devnull.close()
        os.close(self._stdout_dup)
        os.close(self._stderr_dup)
        sys.stdout = self._stdout
        sys.stderr = self._stderr

        return False


def load_cfg(cfg_path: str) -> DotDict:
    """
    Load a YAML config file and return it as a recursive DotDict.
    
    Args:
        cfg_path (str): Path to the YAML config file.
    
    Returns:
        DotDict: Config object with attribute access.
    """
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)

    return DotDict(data)


def load_annotations_scanrefer(ann_path: str, 
                               pkl_path: str) -> Dict[str, dict]:
    """Load ScanRefer annotations and corresponding ScanNet scene data.

    Args:
        ann_path (str): Path to the ScanRefer JSON annotations file.
        pkl_path (str): Path to the .pkl file for ScanNet dataset.

    Returns:
        dict: A dictionary containing:
            - 'annotations' (list[dict]): List of all annotation entries.
            - 'perscene_annotations' (dict): Mapping from scene ID to a list
              of annotations belonging to that scene.
            - 'axis_align_matrices' (dict): Mapping from scene ID to 4x4
              axis-alignment matrices.
            - 'super_points_path' (dict): Mapping from scene ID to the path
              of the superpoint assignments file.
    """
    with open(os.path.join(ann_path), 'r') as f:
        annotations = json.load(f)
    perscene_annotations = {}
    for data in annotations:
        if data['scene_id'] not in perscene_annotations:
            perscene_annotations[data['scene_id']] = []
        perscene_annotations[data['scene_id']].append(data)

    with open(pkl_path, 'rb') as f:
        scannet_data = pickle.load(f)
    axis_align_matrices = {}
    super_points_path = {}
    for elem in scannet_data['data_list']:
        axis_align_matrices[elem['lidar_points']
                                    ['lidar_path'][:-4]] = np.array(
                                        elem['axis_align_matrix'])
        
        sp_pts_mask_path = elem['super_pts_path']
        super_points_path[elem['lidar_points']
                                    ['lidar_path'][:-4]] = sp_pts_mask_path
    
    return dict(
        annotations=annotations,
        perscene_annotations=perscene_annotations,
        axis_align_matrices=axis_align_matrices,
        super_points_path=super_points_path
    )
        
    
def load_annotations_nr3d(ann_path: str, 
                          pkl_path: str) -> Dict[str, dict]:
    """Load Nr3D scene annotations and corresponding ScanNet scene data.

    Args:
        ann_path (str): Path to the Nr3D JSON annotations file.
        pkl_path (str): Path to the .pkl file for ScanNet dataset.

    Returns:
        dict: A dictionary containing:
            - 'perscene_annotations' (dict): Mapping from scene ID to a list
              of annotations belonging to that scene.
            - 'axis_align_matrices' (dict): Mapping from scene ID to 4x4
              axis-alignment matrices.
            - 'super_points_path' (dict): Mapping from scene ID to the path
              of the superpoint assignments file.
    """
    with open(os.path.join(ann_path), 'r') as f:
        perscene_annotations = json.load(f)

    with open(pkl_path, 'rb') as f:
        scannet_data = pickle.load(f)
    axis_align_matrices = {}
    super_points_path = {}
    for elem in scannet_data['data_list']:
        axis_align_matrices[elem['lidar_points']
                                    ['lidar_path'][:-4]] = np.array(
                                        elem['axis_align_matrix'])
        
        sp_pts_mask_path = elem['super_pts_path']
        super_points_path[elem['lidar_points']
                                    ['lidar_path'][:-4]] = sp_pts_mask_path

    return dict(
        perscene_annotations=perscene_annotations,
        axis_align_matrices=axis_align_matrices,
        super_points_path=super_points_path
    )
    

def load_frame_list(frame_path: str, 
                    resolution: Tuple[int, int], 
                    num_images: int) -> Tuple[List, List]:
    """Load images stored in ScanNet format.

    Args:
        frame_path (str): Path to the root directory.
        resolution (tuple[int, int]): Target resolution.
        num_images (int): Number of frames to sample from the full set.

    Returns:
        tuple:
            - list[PIL.Image.Image]: Filtered and resized image frames.
            - list[str]: Filenames of the corresponding filtered frames.
    """
    frame_img_paths = sorted(os.listdir(frame_path))
    frame_jpg_paths = [f for f in frame_img_paths if f.endswith('.jpg')]
    pose_paths = [p.replace('.jpg', '.txt') for p in frame_jpg_paths]
    poses = [np.loadtxt(os.path.join(frame_path, f)) for f in pose_paths]
    poses = [torch.tensor(p, dtype=torch.float32) for p in poses]
    pose_mask = [not torch.isinf(pose).any() for pose in poses]
    frame_list = [Image.open(os.path.join(frame_path, f)) for f in frame_jpg_paths]
    frame_list = [frame.resize((resolution[1], resolution[0])) \
                  for frame in frame_list]
    step = max(1, len(frame_list) // num_images)
    frames = frame_list[::step]
    paths = frame_jpg_paths[::step]
    pose_mask = pose_mask[::step]
    filter_frames = [frame for frame, mask in zip(frames, pose_mask) if mask]
    filter_paths = [path for path, mask in zip(paths, pose_mask) if mask]
    return filter_frames, filter_paths


def load_frame_list_dust3r(frame_path: str, 
                           resolution: Tuple[int, int]) -> Tuple[List, List]:
    """Load posed/unposed images stored in ScanNet format.

    Args:
        frame_path (str): Path to the root directory.
        resolution (tuple[int, int]): Target resolution.

    Returns:
        tuple:
            - list[PIL.Image.Image]: Filtered and resized image frames.
            - list[str]: Filenames of the corresponding filtered frames.
    """
    frame_img_paths = sorted(os.listdir(frame_path))
    frame_jpg_paths = [f for f in frame_img_paths if f.endswith('.jpg')]
    pose_paths = [p.replace('.jpg', '.txt') for p in frame_jpg_paths]
    poses = [np.loadtxt(os.path.join(frame_path, f)) for f in pose_paths]
    poses = [torch.tensor(p, dtype=torch.float32) for p in poses]
    pose_mask = [not torch.isinf(pose).any() for pose in poses]
    frame_list = [Image.open(os.path.join(frame_path, f)) for f in frame_jpg_paths]
    frame_list = [frame.resize((resolution[1], resolution[0])) \
                  for frame in frame_list]
    filter_frames = [frame for frame, mask in zip(frame_list, pose_mask) if mask]
    filter_paths = [path for path, mask in zip(frame_jpg_paths, pose_mask) if mask]
    return filter_frames, filter_paths


def load_scene(root_path: str, 
               scene_id: str, 
               sp_paths: dict, 
               data_prefix=None) -> Dict[str, Optional[np.ndarray]]:
    """Load 3D scene data stored in ScanNet format.

    Args:
        root_path (str): Root directory containing the scene data.
        scene_id (str): Identifier for the scene to load.
        sp_paths (dict): Dictionary mapping scene IDs to superpoint file paths.
        data_prefix (dict, optional): Optional dictionary specifying subfolders
            for different data types. Keys can include 'points', 'instance_mask',
            'semantic_mask', 'super_points'. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - 'point_cloud' (ndarray or None): Nx3 array of point coordinates,
              or None if loading failed.
            - 'instance_mask' (ndarray or None): 1D array of instance IDs,
              or None if loading failed.
            - 'semantic_mask' (ndarray or None): 1D array of semantic labels,
              or None if loading failed.
            - 'sp_pts_mask' (ndarray or None): 1D array of superpoint labels,
              or None if loading failed.
    """
    if not data_prefix:
        data_prefix = {}

    data_prefix_pcd = data_prefix.get('points', 'points')
    try:
        point_cloud = np.fromfile(os.path.join(root_path, data_prefix_pcd, 
                                                scene_id + '.bin'), 
                                    dtype=np.float32).reshape(-1, 6)[:, :3]
    except Exception as e:
        point_cloud = None

    data_prefix_inst = data_prefix.get('instance_mask', 'instance_mask')
    try:
        instance_mask = np.fromfile(os.path.join(root_path, data_prefix_inst, 
                                            scene_id + '.bin'), 
                                dtype=np.int64).reshape(-1)
    except Exception as e:
        instance_mask = None
    
    data_prefix_sem = data_prefix.get('semantic_mask', 'semantic_mask')
    try:
        semantic_mask = np.fromfile(os.path.join(root_path, data_prefix_sem, 
                                            scene_id + '.bin'), 
                                dtype=np.int64).reshape(-1)
    except Exception as e:
        semantic_mask = None
    
    data_prefix_sp_pts = data_prefix.get('super_points', 'super_points')
    try:
        sp_pts_mask = np.fromfile(os.path.join(root_path, data_prefix_sp_pts, 
                                            sp_paths[scene_id]), dtype=np.int64)
    except Exception as e:
        sp_pts_mask = None
    
    return dict(
        point_cloud=point_cloud,
        instance_mask=instance_mask,
        semantic_mask=semantic_mask,
        sp_pts_mask=sp_pts_mask
    )


def align_points(points: np.ndarray, 
                 axis_align_matrix: np.ndarray) -> np.ndarray:
    """Apply transformation defined by 4x4 matrix to point cloud.

    Args:
        points (Tensor or ndarray): Nx3 array of 3D points.
        axis_align_matrix (Tensor or ndarray): 4x4 transformation matrix.

    Returns:
        np.ndarray: Nx3 array of transformed 3D points.
    """
    points = points @ axis_align_matrix[:3, :3].T
    points += axis_align_matrix[:3, -1]
    return points