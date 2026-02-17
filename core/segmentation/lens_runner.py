import torch
import numpy as np
from PIL import Image
from transformers import logging
from qwen_vl_utils import smart_resize
from torchvision.transforms.functional import resize as resize_api
import cv2
from src.open_r1.constants import system_prompt_registry, question_template_registry
logging.set_verbosity_error()

# Running LENS for text-prompted image segmentation.
# This is the modified version of https://github.com/hustvl/LENS/blob/main/demo.py


# ----------- Core functions -------------
def preprocess(image, instruction):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024
    question_template = question_template_registry["pr1_grounding"]
    system_template = system_prompt_registry["default"]
    
    # image = Image.open(image_path).convert(mode="RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height, 
        width, 
        28, 
        max_pixels=1000000
    )
    llm_image = image.resize((resized_width, resized_height))

    sam_image = resize_api(image, (sam_img_size, sam_img_size))
    sam_image = torch.from_numpy(np.array(sam_image)).permute(2, 0, 1).contiguous()
    sam_image = (sam_image - pixel_mean) / pixel_std

    message = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question_template.format(question=instruction)}
        ]},
    ]

    return {
        "image": llm_image,
        "message": message,
        "sam_image": sam_image,
        "ori_hw": (height, width),
        "hw": (resized_height, resized_width)
    }


def evaluate_single(input_data, model, processor):
    message = input_data["message"]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
    image_inputs = [input_data["image"]] * len(texts)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device="cuda", dtype=torch.bfloat16)

    llm_out = model.generate(
        max_new_tokens=256, 
        use_cache=True, 
        do_sample=False, 
        **inputs
    )
    completion_text = processor.batch_decode(llm_out, skip_special_tokens=True)
    new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
    pos = torch.where(llm_out == processor.tokenizer.pad_token_id)
    new_attention_mask[pos] = 0
    inputs.update({"input_ids": llm_out, "attention_mask": new_attention_mask})

    inputs.update({"sam_images": input_data["sam_image"].unsqueeze(0).repeat(len(texts), 1, 1, 1)})
    _, low_res_mask = model(output_hidden_states=True, use_learnable_query=True, **inputs)

    pred_mask = model.postprocess_masks(
        low_res_mask[0],
        orig_hw=input_data["ori_hw"],
    )
    pred_mask = (pred_mask[:, 0] > 0).int()

    return pred_mask, completion_text


def evaluate_batched(input_data, model, processor, device="cuda"):
    """
    Batched inference for a list of input dicts.
    Args:
        input_data: List[Dict] with keys "message", "image", "sam_image", "ori_hw"
        model: multimodal model
        processor: processor with tokenizer/model utilities
        device: device for inference
    Returns:
        pred_masks: List[Tensor]   # One mask tensor per input
        completion_texts: List[str]  # One decoded text output per input
    """

    # Prepare batched text prompts
    texts = [
        processor.apply_chat_template(d["message"], tokenize=False, add_generation_prompt=True)
        for d in input_data
    ]

    # Collect images and SAM images
    images = [d["image"] for d in input_data]
    sam_images = torch.stack([d["sam_image"] for d in input_data]).to(device=device, dtype=torch.bfloat16)

    # Preprocess text+image batch
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device=device, dtype=torch.bfloat16)

    # Generate LLM outputs
    llm_out = model.generate(
        max_new_tokens=256,
        use_cache=True,
        do_sample=False,
        **inputs
    )
    completion_texts = processor.batch_decode(llm_out, skip_special_tokens=True)

    # Attention mask for generated output
    new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
    new_attention_mask[llm_out == processor.tokenizer.pad_token_id] = 0
    inputs.update({"input_ids": llm_out, "attention_mask": new_attention_mask})

    # Add SAM images
    inputs.update({"sam_images": sam_images})

    # Run segmentation inference
    _, low_res_masks = model(
        output_hidden_states=True,
        use_learnable_query=True,
        **inputs
    )

    # Postprocess masks individually
    pred_masks = []
    for i, d in enumerate(input_data):
        mask = model.postprocess_masks(
            low_res_masks[i],
            orig_hw=d["ori_hw"]
        )
        pred_masks.append((mask[:, 0] > 0).int())

    return pred_masks, completion_texts

def rescale_box(pred_box, from_hw, to_hw):
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    scale_w = to_w / from_w
    scale_h = to_h / from_h
    x1, y1, x2, y2 = pred_box
    return [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]


def is_valid_box(box, image_hw=None):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2 or min(x1, y1, x2, y2) < 0:
        return False
    if image_hw:
        h, w = image_hw
        if x2 > w or y2 > h:
            return False
    return True

# ----------- Gradio UI Function -------------
def run_segmentation(image: Image.Image, instruction: str, seg_model, seg_processor):
    image = image.convert("RGB")
    tmp_path = "/tmp/tmp_input.jpg"
    image.save(tmp_path)

    input_data = preprocess(tmp_path, instruction)
    mask, completion_text = evaluate_single(input_data, seg_model, seg_processor)
    mask = mask.squeeze(0).to('cpu').numpy().astype('uint8')
    return mask


def run_segmentation_batched(image_paths: list[str], 
                             instructions: list[str], 
                             seg_model, 
                             seg_processor,
                             frame_resolution,
                             batch_size=1):
    assert len(image_paths) == len(instructions), "images and instructions must be the same length"

    input_data = []
    for img_path, instr in zip(image_paths, instructions):
        img = Image.open(img_path).resize((frame_resolution[1], frame_resolution[0]))
        img = img.convert("RGB")
        data = preprocess(img, instr)
        input_data.append(data)

    # Run batched inference
    all_masks = []
    for batch_idx in range(0, len(input_data), batch_size):
        input_batch = input_data[batch_idx:batch_idx+batch_size]
        masks, _ = evaluate_batched(input_batch, seg_model, seg_processor)
        masks = [m.squeeze(0).to('cpu').numpy().astype('uint8') for m in masks]
        all_masks.extend(masks)
    return all_masks