import torch
from loguru import logger
from datetime import datetime
import os
from terminaltables import AsciiTable
from typing import List, Dict, Union


class ScanReferEvaluator:
    """Evaluator for ScanRefer benchmark.

    This class computes accuracy metrics at multiple IoU thresholds for
    predicted object bounding boxes compared to ground truth.

    Attributes:
        iou_thr (list[float]): IoU thresholds for accuracy computation.
        overall_count (int): Total number of evaluated samples.
        unique_count (int): Number of unique object samples.
        multiple_count (int): Number of multiple-object samples.
        stats_dict (dict): Dictionary storing accuracy counts for overall,
            unique, and multiple categories at each IoU threshold.
        logging_freq (int): Frequency (in steps) at which logging occurs.
    """
    def __init__(self, 
                 iou_thresholds: List[float], 
                 log_dir: str, 
                 logging_freq: int = 100, 
                 suffix: str = '', 
                 write_cfg: bool = False, 
                 cfg_path: str = None):
        """
        Initializes the ScanReferEvaluator.

        Args:
            iou_thresholds (list[float]): List of IoU thresholds.
            log_dir (str): Directory to save log files.
            logging_freq (int, optional): Frequency of logging updates. 
                Defaults to 100.
            suffix (str, optional): Suffix to append to the log filename. 
                Defaults to ''.
            write_cfg (bool, optional): Whether to log the configuration file. 
                Defaults to False.
            cfg_path (str, optional): Path to the configuration file.
                Required only if write_cfg is True.
        """
        self.iou_thr = iou_thresholds
        self.overall_count = 0
        self.unique_count = 0
        self.multiple_count = 0
        self.stats_dict = dict(overall=[0] * len(iou_thresholds), 
                               unique=[0] * len(iou_thresholds),
                               multiple=[0] * len(iou_thresholds))
        
        self.logging_freq = logging_freq
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        log_filename = f"log_{timestamp}{suffix}.log"
        print(f'Log filename: {log_filename}')
        logger.remove() 
        logger.add(os.path.join(log_dir, log_filename), format="{message}")
        if write_cfg:
            with open(cfg_path, "r") as cfg_file:
                logger.info("----- CONFIGURATION FILE -----:")
                logger.info(cfg_file.read())
                logger.info("\n")

    def _axis_aligned_iou_3d(self, 
                             pred: torch.Tensor, 
                             target: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU for axis-aligned 3D bounding boxes.

        Args:
            pred (Tensor): shape (..., 6) — boxes in format [x, y, z, dx, dy, dz]
            target (Tensor): shape (..., 6) — boxes in format [x, y, z, dx, dy, dz]

        Returns:
            Tensor: IoU for each pair, shape (...)
        """
        pred_min = pred[..., :3] - pred[..., 3:] / 2
        pred_max = pred[..., :3] + pred[..., 3:] / 2
        target_min = target[..., :3] - target[..., 3:] / 2
        target_max = target[..., :3] + target[..., 3:] / 2

        inter_min = torch.max(pred_min, target_min)
        inter_max = torch.min(pred_max, target_max)
        inter_dims = torch.clamp(inter_max - inter_min, min=0)
        inter_vol = inter_dims[..., 0] * inter_dims[..., 1] * inter_dims[..., 2]

        pred_vol = (pred[..., 3] * pred[..., 4] * pred[..., 5])
        target_vol = (target[..., 3] * target[..., 4] * target[..., 5])

        union_vol = pred_vol + target_vol - inter_vol
        iou = inter_vol / torch.clamp(union_vol, min=1e-6)

        return iou

    def update(self, pred_bbox, gt_bbox, metadata: dict):
        """
        Updates the evaluator with a predicted and ground truth bounding box.
        
        Args:
            pred_bbox (list or array-like): Predicted bounding box.
            gt_bbox (list or array-like): Ground truth bounding box.
            metadata (dict); Dictionary with additional data.
        """
        iou = self._axis_aligned_iou_3d(torch.tensor(pred_bbox), 
                                        torch.tensor(gt_bbox))
        iou = iou.item()
        
        self.overall_count += 1
        subset = 'multiple' if metadata['is_multiple'] else 'unique'
        if subset == 'multiple':
            self.multiple_count += 1
        else:
            self.unique_count += 1
        for i, thr in enumerate(self.iou_thr):
            if iou > thr:
                self.stats_dict['overall'][i] += 1
                self.stats_dict[subset][i] += 1

        if self.overall_count % self.logging_freq == 0:
            self.compute_metrics(to_log=True)
            

    def compute_metrics(self, 
                        to_log: bool = False) -> Dict[str, Union[int, float]]:
        """
        Computes and prints the accuracy for each IoU threshold.

        Args:
            to_log (bool, optional): If True, logs metrics in a formatted table.
                Defaults to False.

        Returns:
            dict: Dictionary containing:
                - Counts: 'overall_count', 'unique_count', 'multiple_count'
                - Accuracy per IoU threshold for each subset.
        """
        metrics = {}
        subsets = ['overall', 'unique', 'multiple']
        
        metrics['overall_count'] = self.overall_count
        metrics['unique_count'] = self.unique_count
        metrics['multiple_count'] = self.multiple_count

        for i, thr in enumerate(self.iou_thr):
            for subset in subsets:
                curr_subset_count = metrics[f'{subset}_count']
                if curr_subset_count > 0:
                    accuracy = self.stats_dict[subset][i] / curr_subset_count
                else:
                    accuracy = 0.0
                metrics[f"Acc@{thr}_{subset}"] = round(accuracy, 4)

        if to_log:
            table = self._ascii_table_from_dict(metrics)
            logger.info(f'----- Results after {self.overall_count} steps: -----')
            for k, v in metrics.items():
                if '_count' in k:
                    logger.info(f'{k}: {v}')
            logger.info(table)
            logger.info(f'\n')
            
        return metrics
    
    def _ascii_table_from_dict(self, 
                               metrics_dict: Dict[str, Union[int, float]]) -> str:
        """
        Builds an ASCII table from a metrics dictionary.

        Args:
        metrics_dict (dict): Dictionary of metrics.

        Returns:
            str: ASCII-formatted table as a string.
        """
        metric_entries = {k: v for k, v in metrics_dict.items() if k.startswith("Acc")}

        categories = sorted(
            {k.split("_")[1] for k in metric_entries.keys()},
            key=["unique", "multiple", "overall"].index
        )

        metric_types = sorted(
            {k.split("_")[0] for k in metric_entries.keys()},
            key=["Acc@0.25", "Acc@0.5"].index
        )

        table_data = [["Subset"] + metric_types]  # header

        for cat in categories:
            row = [cat]
            for m in metric_types:
                key = f"{m}_{cat}"
                row.append(metric_entries.get(key, ""))
            table_data.append(row)

        table = AsciiTable(table_data)
        table.inner_row_border = True

        return table.table
    

class Nr3DEvaluator:
    """Evaluator for Nr3D benchmark.

        This class tracks accuracy statistics for the Nr3D dataset.

        Attributes:
            overall_count (int): Total number of evaluated samples.
            easy_count (int): Number of easy samples.
            hard_count (int): Number of hard samples.
            view_dep_count (int): Number of view-dependent samples.
            view_indep_count (int): Number of view-independent samples.
            stats_dict (dict): Dictionary storing correct prediction counts
                for each subset.
            logging_freq (int): Frequency (in steps) at which logging occurs.
        """
    def __init__(self, 
                 log_dir, 
                 logging_freq=100, 
                 suffix='', 
                 write_cfg=False, 
                 cfg_path=None):
        """
        Initialize the Nr3DEvaluator.

        Args:
            log_dir (str): Directory where log files will be saved.
            logging_freq (int, optional): Frequency (in steps) at which logging occurs.
                Defaults to 100.
            suffix (str, optional): Suffix appended to the log filename.
                Defaults to ''.
            write_cfg (bool, optional): Whether to log the configuration file.
                Defaults to False.
            cfg_path (str, optional): Path to the configuration file.
                Required only if write_cfg is True.
        """
        self.overall_count = 0
        self.easy_count = 0
        self.hard_count = 0
        self.view_dep_count = 0
        self.view_indep_count = 0
        self.stats_dict = dict(overall=0, easy=0, hard=0, view_dep=0, view_indep=0)
        
        self.logging_freq = logging_freq
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        log_filename = f"log_{timestamp}{suffix}.log"
        print(f'Log filename: {log_filename}')
        logger.remove() 
        logger.add(os.path.join(log_dir, log_filename), format="{message}")
        if write_cfg:
            with open(cfg_path, "r") as cfg_file:
                logger.info("----- CONFIGURATION FILE -----:")
                logger.info(cfg_file.read())
                logger.info("\n")


    def update(self, pred_bbox, gt_bbox, metadata: dict):
        """
        Updates the evaluator with a predicted and ground truth bounding box.
        
        Args:
            pred_bbox (list or array-like): Predicted bounding box.
            gt_bbox (list or array-like): Ground truth bounding box.
            metadata (dict); Dictionary with additional data.
        """
        is_easy = metadata['is_easy']
        is_view_dep = metadata['is_view_dep']
        
        self.overall_count += 1
        subset_easy = 'easy' if is_easy else 'hard'
        subset_view = 'view_dep' if is_view_dep else 'view_indep'
        if subset_easy == 'easy':
            self.easy_count += 1
        else:
            self.hard_count += 1

        if subset_view == 'view_dep':
            self.view_dep_count += 1
        else:
            self.view_indep_count += 1

        if metadata['gt_id'] == metadata['predicted_id']:
            self.stats_dict['overall'] += 1
            self.stats_dict[subset_easy] += 1
            self.stats_dict[subset_view] += 1

        if self.overall_count % self.logging_freq == 0:
            self.compute_metrics(to_log=True)
            

    def compute_metrics(self, to_log=False):
        """
        Computes and prints the accuracy for each subset.

        Args:
            to_log (bool, optional): If True, writes metric values to log.
                Defaults to False.

        Returns:
            dict: Dictionary containing:
                - Counts: 'overall_count', 'easy_count', 'hard_count' etc.
                - Accuracy for each subset.
        """
        metrics = {}
        subsets = ['overall', 'easy', 'hard', 'view_dep', 'view_indep']
        
        metrics['overall_count'] = self.overall_count
        metrics['easy_count'] = self.easy_count
        metrics['hard_count'] = self.hard_count
        metrics['view_dep_count'] = self.view_dep_count
        metrics['view_indep_count'] = self.view_indep_count

        for subset in subsets:
            curr_subset_count = metrics[f'{subset}_count']
            accuracy = self.stats_dict[subset] / curr_subset_count if curr_subset_count > 0 else 0.0
            metrics[f"Acc@{subset}"] = round(accuracy, 4)

        if to_log:
            logger.info(f'----- Results after {self.overall_count} steps: -----')
            for k, v in metrics.items():
                    logger.info(f'{k}: {v}')
            logger.info(f'\n')
            
        return metrics 