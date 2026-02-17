import torch


class Segmentator:
    """Unified wrapper for running image segmentation with different models.

    This class provides a common interface for running segmentation using
    different backends (e.g., LENS or SAM3).
    """
    def __init__(self, cfg, device='cuda'):
        """Initialize the Segmentator.

        Depending on the configuration, this method initializes either the
        LENS model or the SAM3 model.

        Args:
            cfg: Current config.
            device (str, optional): Current device. Defaults to 'cuda'.
        """
        model_name = cfg.seg_model
        if model_name.lower() == 'lens':
            from transformers import AutoProcessor
            from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration_qwen2p5
            from .lens_runner import run_segmentation_batched
            seg_model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(
                cfg.paths.lens_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                ignore_mismatched_sizes=True
            ).to(device)
            seg_processor = AutoProcessor.from_pretrained(cfg.paths.lens_model_path)
            self.seg_model = seg_model
            self.seg_processor = seg_processor
            self.seg_runner = run_segmentation_batched
        
        elif model_name.lower() == 'sam3':
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            from .sam_runner import run_segmentation_batched
            bpe_path = f"{cfg.paths.sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
            model = build_sam3_image_model(bpe_path=bpe_path, 
                                        checkpoint_path=cfg.paths.sam3_model_path)
            seg_processor = Sam3Processor(model, confidence_threshold=0.5)
            self.seg_processor = seg_processor
            self.seg_runner = run_segmentation_batched
        
        else:
            raise NotImplementedError('Model name should be either LENS or sam3!')
        
        self.model_name = model_name.lower()
    
    def run_segmentation(
        self, 
        image_paths: list[str],
        instructions: list[str],
         **kwargs
    ):
        """Run segmentation on a set of images with text prompts.

        Args:
            image_paths (list[str]): List of image file paths.
            instructions (list[str]): List of text prompts.
            **kwargs: Additional keyword arguments.
        """
        if self.model_name == 'lens':
            return self.seg_runner(
                image_paths=image_paths,
                instructions=instructions,
                seg_model=self.seg_model,
                seg_processor=self.seg_processor,
                **kwargs
            )
        
        elif self.model_name == 'sam3':
            return self.seg_runner(
                image_paths_arr=image_paths,
                instructions_arr=instructions,
                seg_processor=self.seg_processor,
                **kwargs
            )
        
        else:
            raise NotImplementedError('Model name should be either LENS or sam3!')