import torch
import os
import numpy as np
import json
import tqdm
import warnings
warnings.filterwarnings("ignore")
import argparse
from datetime import datetime
from core.clip_retrieval import ClipRetrieval
from core.vlm_filtering import vlm_filtering
from core.segmentation.segmentator import Segmentator
from core.mv_aggregation import mv_aggregation
from utils.grounding_utils import *
from utils.evaluate import ScanReferEvaluator
from utils.mask_utils import *
from utils.img2pointcloud import multiple_img2pointcloud
from utils.scanrefer_utils import get_inst_predictions, gt_bbox_from_instances


def init_Z3D(cfg, args):
    retriever = ClipRetrieval(model_name=cfg.clip_params.model_name,
                            pretrained=cfg.clip_params.pretrained,
                            batch_size=cfg.clip_params.clip_bs)

    vlm_config = {
            "provider": "vllm",
            "model": cfg.vlm_model,
            'llm_server_url': f"http://0.0.0.0:{args.port}/v1" ,
            'api_key': 42,
            'name': cfg.vlm_name
        }
    
    segmentator = Segmentator(cfg)

    ann_dict = load_annotations_scanrefer(ann_path=cfg.paths.ann_path, 
                                          pkl_path=cfg.paths.pkl_path)                        
    iou_thr = [0.25, 0.5]
    log_dir_path = os.path.join(cfg.log_params.log_dir, cfg_folder)
    os.makedirs(log_dir_path, exist_ok=True)
    suff = f'_{cfg.inst_pred_params.backbone}_score_thr={cfg.inst_pred_params.score_thr}'
    evaluator = ScanReferEvaluator(iou_thr, 
                                log_dir=log_dir_path,  
                                logging_freq=cfg.log_params.print_freq, 
                                suffix=suff, 
                                write_cfg=True,
                                cfg_path=cfg_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    AGENT_OUTPUT_DIR = f"{cfg.paths.output_agent_dir}_{timestamp}"
    RESULT_DIR = os.path.join('results', cfg_folder, timestamp)
    os.makedirs(RESULT_DIR, exist_ok=False)

    return dict(
        ann_dict=ann_dict,
        frame_retriever=retriever,
        vlm_model_cfg=vlm_config,
        segmentator=segmentator,
        evaluator=evaluator,
        agent_output_dir=AGENT_OUTPUT_DIR,
        result_save_dir=RESULT_DIR
    )


def run_single_scene(
        cfg,
        scene_id,
        scene_ann,
        frame_retriever,
        vlm_model_cfg,
        segmentator,
        evaluator,
        agent_output_dir,
        result_save_dir,
        metadata
    ):
    image_root = os.path.join(cfg.paths.root_path, 'posed_images', scene_id)
    RESOLUTION = metadata['frame_resolution']
    frames, jpg_paths = load_frame_list(image_root, 
                                        resolution=RESOLUTION, 
                                        num_images=cfg.eval_params.num_images)  
    frame_retriever.set_frames(frames, normalize=True)
    axis_align_matrices = metadata['axis_align_matrices']
    super_points_path = metadata.get('super_points_path', '')
    scene = load_scene(cfg.paths.root_path, 
                       scene_id=scene_id, sp_paths=super_points_path)
    point_cloud = scene['point_cloud']
    instance_mask = scene['instance_mask']
    semantic_mask = scene['semantic_mask']
    sp_pts_mask = scene['sp_pts_mask']
    axis_align_matrix = axis_align_matrices[scene_id]
    inst_predictions_3d = get_inst_predictions(scene_id=scene_id,
                                    method=cfg.inst_pred_params.backbone,
                                    args=dict(
                                        olt_root_path=cfg.paths.olt_root_path,
                                        score_thr=cfg.inst_pred_params.score_thr,
                                        axis_align_matrix=axis_align_matrix,
                                        point_cloud=point_cloud
                                    ))

    final_results_to_save = []
    gt_bboxes = []
    prompt_arr = []
    is_multiple = []
    for item in scene_ann:     
        gt_bbox, is_item_multiple = gt_bbox_from_instances(
            int(item['object_id']),
            point_cloud,
            semantic_mask,
            instance_mask,
            axis_align_matrix
        )
        is_multiple.append(is_item_multiple)
        gt_bboxes.append(gt_bbox)

        prompt = item['description']
        prompt_arr.append(prompt)        

    #Frame retrieval with CLIP
    chosen_frames_clip, frames_indices = frame_retriever.query_topk(prompt_arr, 
                                                             topk=cfg.topk_clip, 
                                                             normalize=True)
    #Frame filtering with VLM
    chosen_frames, topk_indices, vlm_scores = (
        vlm_filtering(
            chosen_frames_clip,
            frames_indices,
            prompt_arr,
            vlm_model_cfg,
            topk=cfg.topk,
            n_parallel_queries=cfg.vlm_parallel_queries,
            metadata=dict(
                image_root=image_root,
                jpg_paths=jpg_paths
            )
        )
    )

    score_mask = vlm_scores >= cfg.vlm_score_thr
    score_mask[:, 0] = True  # always keep first frame

    # Build filtered frames & indices for segmentation
    filtered_frames = []
    filtered_jpg_idxs = []
    filtered_jpg_paths = []
    for i in range(len(prompt_arr)):
        mask = score_mask[i]
        curr_filtered_frames = [chosen_frames[i][m] for m in range(cfg.topk) if mask[m]]

        curr_filtered_idxs = [int(jpg_paths[topk_indices[i][m]][:-4]) 
                         for m in range(cfg.topk) if mask[m]]
        
        curr_filtered_paths = [os.path.join(image_root, str(path).zfill(5) + '.jpg') 
                          for path in curr_filtered_idxs]
        
        filtered_frames.append(curr_filtered_frames)
        filtered_jpg_idxs.append(curr_filtered_idxs)
        filtered_jpg_paths.append(curr_filtered_paths)

    masks = []
    flat_paths = [path for path_list in filtered_jpg_paths for path in path_list]
    flat_prompts = [prompt_arr[ii] for ii, frames_list 
                    in enumerate(filtered_frames) for _ in frames_list]

    seg_model_name = cfg.seg_model.lower()
    if seg_model_name == 'lens':
        kwargs = {
            'frame_resolution': RESOLUTION,
            'batch_size': cfg.seg_bs
        }
    elif seg_model_name == 'sam3':
        placeholder = np.zeros(RESOLUTION)
        output_agent_dir=f'{agent_output_dir}/{scene_id}'
        kwargs = {
            'llm_config': vlm_model_cfg,
            'output_agent_dir': output_agent_dir,
            'placeholder': placeholder,
            'max_parallel_queries': cfg.vlm_parallel_queries
        }
    
    # Run segmentation
    curr_masks = segmentator.run_segmentation(
        flat_paths,
        flat_prompts,
        **kwargs
    )

    # Reshape masks back
    idx = 0
    for frames_list in filtered_frames:
        n = len(frames_list)
        masks.append(curr_masks[idx: idx + n])
        idx += n

    for idx in range(len(prompt_arr)):
        curr_gt = gt_bboxes[idx]
        frame_jpg_ids = filtered_jpg_idxs[idx]
        frame_masks = masks[idx]

        # Project 2D masks to 3D
        _, mask_3d_list = multiple_img2pointcloud(
            cfg.paths.root_path,
            scene_id,
            frame_jpg_ids,
            desired_resolution=RESOLUTION,
            masks=[[m] for m in frame_masks],
            write=False
        )
            
        # Project 3D masks onto the point cloud with 1-NN
        final_bin_masks = project_masks2pcd(mask_3d_list, point_cloud)

        # Multi-view aggregation
        best_mask_id = mv_aggregation(final_bin_masks, 
                                       inst_predictions_3d, 
                                       point_cloud)
        
        if best_mask_id is not None:
            ids = inst_predictions_3d[best_mask_id]
            final_mask_3d = point_cloud[ids]
            N = point_cloud.shape[0]
            final_bin_mask = np.zeros(N)
            final_bin_mask[ids] = 1
        else: ## if not matched, give mask from first frame
            if len(final_bin_masks) > 0:
                final_bin_mask = final_bin_masks[0].astype('int')
                final_mask_3d = point_cloud[final_bin_mask.astype('bool')]
            else:
                final_mask_3d = None

        try:   
            final_mask_3d = align_points(final_mask_3d, axis_align_matrix)
            final_mask_3d = torch.tensor(final_mask_3d)
            if cfg.use_sp_pts:
                aligned_pc = torch.tensor(align_points(point_cloud, axis_align_matrix))
                final_mask_3d = trim_mask_by_superpoints(
                    torch.tensor(final_bin_mask).bool(),
                    torch.tensor(sp_pts_mask),
                    aligned_pc
                )
                xyz_min = final_mask_3d.min(dim=0).values
                xyz_max = final_mask_3d.max(dim=0).values
            else:
                xyz_min = torch.quantile(final_mask_3d, q=0.01, dim=0)
                xyz_max = torch.quantile(final_mask_3d, q=0.99, dim=0)
            
            center = (xyz_max + xyz_min) / 2
            size = xyz_max - xyz_min
            pred_bbox = np.hstack((center, size))

        except Exception as e:
            gt_center = curr_gt[:3]
            gt_size = curr_gt[3:]
            pred_bbox = np.hstack((gt_center + 1e8, gt_size)) #this bbox always yields 0 IoU with gt

        metadata = dict(is_multiple=is_multiple[idx])
        evaluator.update(pred_bbox, curr_gt, metadata=metadata)
        result_dict = {
            'query': prompt_arr[idx],
            'gt_bbox': curr_gt.tolist(),
            'pred_bbox': pred_bbox.tolist(),
            'unique': not is_multiple[idx]
        }
        final_results_to_save.append(result_dict)

    json_string = json.dumps(final_results_to_save, indent=4)
    with open(os.path.join(result_save_dir, scene_id + '.json'), 'w') as f:
        f.write(json_string)

    return evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="VLLM server port"
    )

    args = parser.parse_args()

    cfg_name = args.config
    cfg_folder = os.path.splitext(os.path.basename(cfg_name))[0]
    cfg = load_cfg(cfg_name)

    init_dict = init_Z3D(cfg, args)

    frame_retriever = init_dict['frame_retriever']
    vlm_model_cfg = init_dict['vlm_model_cfg']
    segmentator = init_dict['segmentator']
    evaluator = init_dict['evaluator']
    agent_output_dir = init_dict['agent_output_dir']
    result_save_dir = init_dict['result_save_dir']
    ann_dict = init_dict['ann_dict']

    perscene_annotations = ann_dict['perscene_annotations']
    for scene_id, scene_ann in tqdm.tqdm(list(perscene_annotations.items()), ncols=100):
        evaluator = run_single_scene(
            cfg,
            scene_id,
            scene_ann,
            frame_retriever=frame_retriever,
            vlm_model_cfg=vlm_model_cfg,
            segmentator=segmentator,
            evaluator=evaluator,
            agent_output_dir=agent_output_dir,
            result_save_dir=result_save_dir,
            metadata=dict(
                frame_resolution=(480, 640),
                axis_align_matrices=ann_dict['axis_align_matrices'],
                super_points_path=ann_dict['super_points_path']
            )
        )
        if not cfg.eval_params.full_eval and evaluator.overall_count > cfg.eval_params.num_samples:
            break

    metrics = evaluator.compute_metrics(to_log=True)
    print('----- FINAL EVALUATION RESULTS -----')
    print(f'Number of images: {cfg.eval_params.num_images}')
    table = evaluator._ascii_table_from_dict(metrics)
    for k, v in metrics.items():
        if '_count' in k:
            print(f'{k}: {v}')
    print(table)