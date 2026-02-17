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
from utils.evaluate import Nr3DEvaluator
from utils.mask_utils import *
from utils.img2pointcloud import multiple_img2pointcloud
from utils.nr3d_utils import get_instances_3d


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

    ann_dict = load_annotations_nr3d(ann_path=cfg.paths.ann_path, 
                                          pkl_path=cfg.paths.pkl_path)                      
    log_dir_path = os.path.join(cfg.log_params.log_dir, cfg_folder)
    os.makedirs(log_dir_path, exist_ok=True)
    evaluator = Nr3DEvaluator(log_dir=log_dir_path,  
                          logging_freq=cfg.log_params.print_freq, 
                          suffix=f'', 
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
    axis_align_matrix = axis_align_matrices[scene_id]
    
    instances_3d, gt_bbox_ids, gt_bbox_list, gt_bbox_lookup_table  = (
        get_instances_3d(
            scene_ann, 
            point_cloud,
            axis_align_matrix
        )
    )

    scene_items = scene_ann['items']
    final_results_to_save = []
    gt_bboxes = []
    gt_ids = []
    prompt_arr = []
    is_easy = []
    is_view_dep = []
    for item in scene_items:
        gt_bboxes.append(item['gt_bbox'])
        gt_ids.append(item['gt_id'])
        prompt_arr.append(item['query'])       
        is_easy.append(item['easy'])
        is_view_dep.append(item['view_dep'])

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
    score_mask[:, 0] = True # always keep first frame

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
        best_mask_id  = mv_aggregation(final_bin_masks, 
                                       instances_3d, 
                                       point_cloud)

        # if can't match, try to find nearest bbox
        if best_mask_id is None:
            try:
                bin_mask = final_bin_masks[0]
                aligned_pc = align_points(point_cloud, 
                                          axis_align_matrix=axis_align_matrix)
                mask_3d = aligned_pc[bin_mask.astype('int').astype('bool')]
                xyz_min = np.quantile(mask_3d, q=0.05, axis=0, keepdims=True)
                xyz_max = np.quantile(mask_3d, q=0.95, axis=0, keepdims=True)
                mask_center = (xyz_max + xyz_min) / 2
                bboxes_np = np.array(gt_bbox_list)
                dist_array = np.square(bboxes_np[..., :3] - mask_center)
                bbox_idx = np.argmin(dist_array.sum(axis=-1))
                best_mask_id = gt_bbox_ids[bbox_idx]
            except Exception as e:
                best_mask_id = list(gt_bbox_lookup_table.keys())[0]
        
        pred_bbox = gt_bbox_lookup_table[best_mask_id]
        metadata = dict(is_easy=is_easy[idx], 
                        is_view_dep=is_view_dep[idx], 
                        gt_id=gt_ids[idx], 
                        predicted_id=best_mask_id)
        evaluator.update(pred_bbox, curr_gt, metadata=metadata)

        result_dict = {
            'query': prompt_arr[idx],
            'gt_id': metadata['gt_id'],
            'predicted_id': metadata['predicted_id'],
            'gt_bbox': curr_gt,
            'pred_bbox': pred_bbox,
            'easy': metadata['is_easy'],
            'view_dep': metadata['is_view_dep']
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
    for k, v in metrics.items():
        print(f'{k}: {v}')