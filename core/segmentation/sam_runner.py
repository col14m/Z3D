import torch
import os
from functools import partial
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference
import pycocotools.mask as mask_utils
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from utils.grounding_utils import SilentAllOutput
import time
from typing import List, Optional


def decode_masks(output_dict: dict) -> Optional[np.ndarray]:
    """Decode RLE-encoded segmentation masks to binary masks.

    Args:
        output_dict (dict): Dictionary containing model outputs, including
            original image height (``orig_img_h``), original image width
            (``orig_img_w``), and RLE-encoded predicted masks
            (``pred_masks``).

    Returns:
        ndarray or None: Decoded binary mask corresponding to the prediction 
        with highest score, or None if no masks are present.
    """
    orig_h = int(output_dict["orig_img_h"])
    orig_w = int(output_dict["orig_img_w"])
    rle_masks = [
           {"size": (orig_h, orig_w), "counts": rle}
           for rle in output_dict["pred_masks"]
           ]
    binary_masks = [mask_utils.decode(rle) for rle in rle_masks]
    return binary_masks[0] if len(binary_masks) > 0 else None


def run_segmentation(image_path: str, 
                     instruction: str, 
                     seg_processor, 
                     llm_config: dict, 
                     output_agent_dir: str) -> tuple[Optional[np.ndarray], float]:
    """Run segmentation of single image with SAM3 Agent.

    Args:
        image_path (str): Path to the input image.
        instruction (str): Text instruction describing the target object.
        seg_processor: Segmentation processor model.
        llm_config (dict): Configuration dictionary for the VLM model.
        output_agent_dir (str): Directory for agent outputs.

    Returns:
        tuple: A tuple containing:
            - ndarray or None: Decoded binary segmentation mask, or None
              if no mask is produced.
            - float: Total inference time in seconds.
    """
    start_perf = time.perf_counter()
    image_path = os.path.abspath(image_path)
    LLM_SERVER_URL = llm_config['llm_server_url']
    send_generate_request = partial(send_generate_request_orig, 
                                    server_url=LLM_SERVER_URL, 
                                    model=llm_config["model"], 
                                    api_key=llm_config["api_key"])
    call_sam_service = partial(call_sam_service_orig, 
                               sam3_processor=seg_processor)

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        final_output_dict = run_single_image_inference(
                image_path, instruction, llm_config, send_generate_request, 
                call_sam_service, debug=False, output_dir=output_agent_dir
        )
    mask = decode_masks(final_output_dict)
    elapsed_time = time.perf_counter() - start_perf
    return mask, elapsed_time


async def run_segmentation_async(
    image_path: str,
    instruction: str,
    seg_processor,
    llm_config: dict,
    output_agent_dir: str,
    executor: ThreadPoolExecutor,   
) -> tuple[Optional[np.ndarray], float]:
    """Run image segmentation asynchronously using a thread pool executor.

    Args:
        image_path (str): Path to the input image.
        instruction (str): Text instruction describing the target object.
        seg_processor: Segmentation processor model.
        llm_config (dict): Configuration dictionary for the VLM model.
        output_agent_dir (str): Directory for agent outputs.
        executor (ThreadPoolExecutor): Executor object.

    Returns:
        tuple: A tuple containing:
            - ndarray or None: Decoded binary segmentation mask, or None
              if no mask is produced.
            - float: Total inference time in seconds.
    """
    loop = asyncio.get_running_loop()

    func = partial(
        run_segmentation,
        image_path,
        instruction,
        seg_processor,
        llm_config,
        output_agent_dir,
    )

    return await loop.run_in_executor(executor, func)


def run_segmentation_batched(
    image_paths_arr: List[str],
    instructions_arr: List[str],
    seg_processor,
    llm_config: dict,
    output_agent_dir: str,
    placeholder: np.ndarray, 
    max_parallel_queries: int = 64
) -> tuple[list[np.ndarray], list[float]]:
    """Run multiple image segmentation with asynchronous execution.

    Args:
        image_paths_arr (list[str]): List of paths to input images.
        instructions_arr (list[str]): List of text queries.
        seg_processor: Segmentation model.
        llm_config (dict): Configuration for the VLM model.
        output_agent_dir (str): Directory for agent outputs.
        placeholder (ndarray): Placeholder mask returned when segmentation
            fails after several retries.
        max_parallel_queries (int): Maximum number of parallel segmentation
            requests. Defaults to 64.

    Returns:
        tuple: A tuple containing:
            - list[np.ndarray]: List of binary segmentation masks (``uint8``),
              one per input image.
            - list[float]: List of inference times for successful requests.
    """

    request_arr_size = len(image_paths_arr)

    executor = ThreadPoolExecutor(max_workers=max_parallel_queries)

    timing_stats = []

    async def _run_batch_async():
        async def safe_run(img: str, instr: str, output_dir: str):
            """
            Runs run_segmentation_async safely with one retry.
            Returns placeholder if both attempts fail.
            """
            try:
                result, elapsed_time = await run_segmentation_async(
                    img, instr, seg_processor, 
                    llm_config, output_dir, executor
                )
                timing_stats.append(elapsed_time) 
                return result
            except Exception as e1:
                try:
                    result, elapsed_time = await run_segmentation_async(
                        img, instr, seg_processor, 
                        llm_config, output_dir + '_attempt_2', executor
                    )
                    timing_stats.append(2 * elapsed_time) 
                    return result
                except Exception as e2:
                    return placeholder.copy()
                
        output_folders = [os.path.join(output_agent_dir, f'sample_{i}') \
                          for i in range(request_arr_size)]
        tasks = [safe_run(img, instr, out_dir) for img, instr, out_dir \
                 in zip(image_paths_arr, instructions_arr, output_folders)]
        return await asyncio.gather(*tasks)

    with SilentAllOutput(): # this thing suppresses all console prints of SAM3 agent
        results = asyncio.run(_run_batch_async())
    executor.shutdown(wait=True)
    for i in range(len(results)):
        if results[i] is None:
            results[i] = placeholder.copy()
    results = [m.astype('uint8') for m in results]
    return results