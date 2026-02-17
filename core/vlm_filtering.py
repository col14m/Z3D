import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import re
from typing import Union, Optional
from sam3.agent.client_llm import send_generate_request
import time
import torch
import os
import PIL


def parse_qwen_think_response(
    response: Union[str, list[str]], 
    parse_reasoning: bool
) -> dict[str, Optional[Union[int, str]]]:
    """
    Qwen parser with <think>...</think> reasoning template.

    Logic:
    1) Extract reasoning between <think> and </think> (if exists)
    2) Cut everything up to and including the FIRST </think>
    3) From the remaining text:
        - extract first integer as score
        - keep clean_text

    Returns:
        {
            "score": int | None,
            "reasoning": str | None,
            "clean_text": str,
            "raw": str
        }
    """

    if isinstance(response, list):
        raw = "\n".join([r for r in response if r])
    else:
        raw = response or ""

    text = raw.strip()

    reasoning = None
    clean_text = text
    
    if not parse_reasoning:
        clean_text = text
        reasoning = None
    else:
        close_match = re.search(r"</think>", text, flags=re.IGNORECASE)
        if close_match:
            reasoning = text[:close_match.start()]
            after = text[close_match.end():]
            clean_text = after.strip()
        else:
            return {
                "score": None,
                "reasoning": None,
                "clean_text": None,
                "raw": raw,
            }


    score = None
    m_score = re.search(r"-?\d+", clean_text)
    if m_score:
        try:
            score = int(m_score.group())
        except Exception:
            score = None

    return {
        "score": score,
        "reasoning": reasoning,
        "clean_text": clean_text,
        "raw": raw,
    }


async def run_generate_request_async(messages: list[dict], 
                                     server_url: str, 
                                     model: str, 
                                     api_key: str, 
                                     max_tokens: int, 
                                     executor: ThreadPoolExecutor):
    """Run a synchronous generation request asynchronously.

    Args:
        messages (list[dict]): List of message payloads to be sent to the
            generation API.
        server_url (str): URL of the inference server.
        model (str): Model name to be used for generation.
        api_key (str): API key for authentication.
        max_tokens (int): Maximum number of tokens to generate.
        executor (Executor): Executor used to run the synchronous request
            in a separate thread or process.

    Returns:
        Any: Raw response returned by the synchronous
        ``send_generate_request`` function.
    """
    loop = asyncio.get_running_loop()
    func = partial(send_generate_request, messages, server_url, 
                   model, api_key, max_tokens)
    return await loop.run_in_executor(executor, func)


def inference_images_batch_vllm(
    images: list[str],              
    prompts: list[str],
    server_url: str,
    model: str,
    parse_reasoning: bool,
    api_key: str = None,
    max_tokens: int = 4096,
    max_parallel_queries: int = 32
):
    """Run asynchronous image–text inference using a vLLM server.

    Args:
        images (list[str]): List of image paths.
        prompts (list[str]): List of text prompts corresponding to each image.
        server_url (str): URL of the vLLM inference server.
        model (str): Model name to be used for generation.
        parse_reasoning (bool): Whether to parse reasoning content
            from the model output.
        api_key (str, optional): API key for authentication. Defaults to None.
        max_tokens (int): Maximum number of tokens to generate per request.
            Defaults to 4096.
        max_parallel_queries (int): Maximum number of concurrent requests. 
            Defaults to 32.

    Returns:
        tuple: A tuple containing:
            - list[int | None]: Parsed relevance scores for each image.
            - list[str]: Cleaned text outputs for each request.
            - list[float]: Inference time (in seconds) for each request.
    """

    assert len(images) == len(prompts), "images and prompts must match"
    batch_size = len(images)
    timings = []
    executor = ThreadPoolExecutor(max_workers=max_parallel_queries)

    async def _run_all_async():
        scores = [None] * batch_size
        clean_texts = [None] * batch_size

        async def worker(i, img_path, prompt):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
            start_perf = time.perf_counter()
            raw_answer = await run_generate_request_async(
                messages, server_url, model, api_key, max_tokens, executor
            )

            parsed = parse_qwen_think_response(raw_answer, parse_reasoning)

            if parsed["score"] is None:
                raw_answer = await run_generate_request_async(
                    messages, server_url, model, api_key, max_tokens, executor
                )
                parsed = parse_qwen_think_response(raw_answer, parse_reasoning)
                if parsed["score"] is None:
                    parsed["score"] = -1

            scores[i] = parsed["score"]
            clean_texts[i] = parsed["clean_text"]
            elapsed_time = time.perf_counter() - start_perf
            timings.append(elapsed_time)

        tasks = [
            asyncio.create_task(worker(i, img, prompt))
            for i, (img, prompt) in enumerate(zip(images, prompts))
        ]
        await asyncio.gather(*tasks)
        return scores, clean_texts

    scores, texts = asyncio.run(_run_all_async())
    executor.shutdown(wait=True)
    return scores, texts, timings


def stable_topk(x: torch.Tensor, 
                k: int, 
                dim: int = -1, 
                largest: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k elements along a dimension with stable ordering.

    Args:
        x (Tensor): Input tensor.
        k (int): Number of elements to select.
        dim (int, optional): Dimension along which to perform top-k
            selection. Defaults to -1.
        largest (bool, optional): If True, select the k largest values.
            If False, select the k smallest values. Defaults to True.

    Returns:
        tuple[Tensor, Tensor]:
            - Tensor: Top-k values along the specified dimension.
            - Tensor: Indices of the selected values in the original tensor.
    """
    if largest:
        sorted_vals, sorted_idx = torch.sort(-x, dim=dim, stable=True)
        sorted_vals = -sorted_vals
    else:
        sorted_vals, sorted_idx = torch.sort(x, dim=dim, stable=True)

    dim_size = x.size(dim)
    k = min(k, dim_size)

    indices = torch.arange(0, k, device=x.device)

    top_vals = sorted_vals.index_select(dim, indices)
    top_idx  = sorted_idx.index_select(dim, indices)

    return top_vals, top_idx


def vlm_filtering(
    chosen_frames: list[list[PIL.Image]],
    frames_idxs: list[list[int]],
    query_arr: list[str],
    model_cfg: dict,
    topk: int,
    n_parallel_queries: int,
    metadata: dict
):
    """Filter retrieved frames using a VLM-based scoring.

    Args:
        chosen_frames (list[list[PIL.Image]]): List of frame lists, where each
            inner list contains candidate frames for a single query.
        frames_idxs (list[list[int]]): Indices of the candidate frames
            corresponding to `chosen_frames`.
        query_arr (list[str]): List of queries, one per group of frames.
        model_cfg (dict): Configuration for the VLM.
        topk (int): Number of top-scoring frames to keep per query.
        n_parallel_queries (int): Maximum number of parallel VLM queries.
        metadata (dict): Dictionary with image metadata, including:
            - 'image_root': Root directory of images.
            - 'jpg_paths': List of image file paths indexed by frame ID.

    Returns:
        tuple:
            - list[list[PIL.Image]]: Filtered top-k frames per query.
            - Tensor: Indices of the selected top-k frames
              (shape: [num_queries, topk]).
            - Tensor: VLM relevance scores for the selected frames
              (shape: [num_queries, topk]).
    """
    filtered_frames = []
    topk_indices = torch.zeros((frames_idxs.shape[0], topk), dtype=torch.long)
    vlm_scores = torch.zeros((frames_idxs.shape[0], topk), dtype=torch.long)
    flat_indices = [ind for ind_list in frames_idxs for ind in ind_list]
    image_root = metadata['image_root']
    jpg_paths = metadata['jpg_paths']
    flat_image_paths = [os.path.join(image_root, jpg_paths[ind]) 
                        for ind in flat_indices]
    flat_prompts_arr = [generate_scoring_prompt(query_arr[ii]) for ii, frames_list 
                        in enumerate(chosen_frames) for _ in frames_list]
    model_txt = model_cfg['model'].lower()
    name_txt = model_cfg['name'].lower() 
    parse_reasoning = 'thinking' in model_txt or 'thinking' in name_txt

    flat_scores, _, _= inference_images_batch_vllm(
            flat_image_paths,
            flat_prompts_arr,
            server_url=model_cfg['llm_server_url'],
            model=model_cfg['model'],
            parse_reasoning=parse_reasoning,
            api_key=model_cfg['api_key'],
            max_parallel_queries=n_parallel_queries
        )
    
    vsc = []
    idx = 0
    for frames_list in chosen_frames:
        n = len(frames_list)
        vsc.append(flat_scores[idx: idx + n])
        idx += n
    vsc = torch.tensor(vsc)
    _, idxs = stable_topk(vsc, k=topk, dim=-1)
    for k in range(len(chosen_frames)):
        idxs_k = idxs[k]
        topk_indices[k] = frames_idxs[k, idxs_k]
        vlm_scores[k] = vsc[k, idxs_k]
        filtered_frames.append([chosen_frames[k][i] for i in idxs_k])
    
    return filtered_frames, topk_indices, vlm_scores


def generate_scoring_prompt(template: str) -> str:
    """Generate a prompt for VLM-based image scoring with textual description.

    Args:
        template (str): Textual description

    Returns:
        str: A formatted prompt instructing a model to score the relevance
        between an image and the provided textual description.
    """
    return f"""
        Instruction:
        You are given an image and a textual description. Description contains the information for the visual grounding: target object description and its spatial relationships to neighbour objects.
        Your task is to evaluate how relevant the description is to the content of the image.

        Text description:
        {template}

        Relevance scoring scale:
        - 0  – The target object does not appear in the image.
        - 2  – Only a very small part of the target object is visible in the image.
        - 4  – The target object is clearly present, but the spatial relationships are completely incorrect.
        - 6  – Most of the target object is visible, and some spatial relationships are correct.
        - 8  – The target object is fully or almost fully visible, and most spatial relationships are correct.
        - 10 – The description perfectly matches the content of the image (100% agreement).

        Output format (MUST follow exactly):
        RELEVANCE_SCORE: <score>

        Where <score> is one of: 0, 2, 4, 6, 8, 10.
        Do not output anything else. No explanation of your decision, no other words. Just follow the output format.

        Task:
        Based on the image and the provided description, determine the correct relevance score and output it in the required format.
        """