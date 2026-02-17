import torch
import open_clip
from typing import List
import PIL


class ClipRetrieval:
    def __init__(self, 
                 model_name: str, 
                 pretrained: str, 
                 batch_size: int = 16, 
                 device: str = 'cuda'):
        """Initialize a CLIP-based image–text retrieval module.

        Args:
            model_name (str): Name of the CLIP model architecture.
            pretrained (str): Identifier of the pretrained weights to load.
            batch_size (int): Batch size used for image and text encoding.
                Defaults to 16.
            device (str): Device to run the model on (e.g., ``'cuda'`` or
                ``'cpu'``). Defaults to ``'cuda'``.
        """
        self.device = device
        clip_model, _, clip_processor = (
            open_clip.create_model_and_transforms(model_name, 
                                                  pretrained=pretrained)
        )
        
        clip_model = clip_model.to(device)
        clip_model.eval()
        clip_tokenizer = open_clip.get_tokenizer(model_name)

        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.clip_tokenizer = clip_tokenizer
        self.image_features = None
        self.frame_list = None
        self.batch_size = batch_size
    
    @torch.no_grad()
    def set_frames(self, 
                   frame_list: list[PIL.Image], 
                   normalize: bool = False) -> None:
        """Precompute image embeddings for a list of frames.

        Args:
            frame_list (list): List of input frames to be encoded.
            normalize (bool): Whether to L2-normalize the image embeddings.

        Returns:
            None: Image embeddings are stored in ``self.image_features`` and
            frames are stored in ``self.frame_list``.
        """
        self.frame_list = frame_list
        image_features = []
        preprocessed_images = [self.clip_processor(frame) for frame in frame_list]
        preprocessed_images = torch.stack(preprocessed_images).to(self.device)

        for i in range(0, len(frame_list), self.batch_size):
            end_idx = min(i + self.batch_size, len(frame_list))
            curr_batch = preprocessed_images[i:end_idx]
            batch_image_features = self.clip_model.encode_image(curr_batch)
            if normalize:
                denom = batch_image_features.norm(dim=-1, keepdim=True)
                batch_image_features = batch_image_features / denom
            image_features.append(batch_image_features)

        self.image_features = torch.cat(image_features, dim=0).to(self.device)

    @torch.no_grad()
    def _get_text_features(self, 
                           prompts: List[str], 
                           normalize: bool = False) -> torch.Tensor:
        """Encode text prompts into feature embeddings.

        Args:
            prompts (list[str]): List of text prompts to be encoded.
            normalize (bool): Whether to L2-normalize the text embeddings.

        Returns:
            Tensor: Encoded text feature embeddings for all prompts.
        """
        text_features = []

        for i in range(0, len(prompts), self.batch_size):
            end_idx = min(i + self.batch_size, len(prompts))
            batch_prompts = prompts[i:end_idx]
            tok_batch = self.clip_tokenizer(batch_prompts).to(self.device)
            batch_text_features = self.clip_model.encode_text(tok_batch)
            if normalize:
                denom = batch_text_features.norm(dim=-1, keepdim=True)
                batch_text_features = batch_text_features / denom
            text_features.append(batch_text_features)

        text_features = torch.cat(text_features, dim=0).to(self.device)
        return text_features
        
    def query(self, 
              prompts: List[str], 
              normalize: bool = False) -> tuple[list[PIL.Image], torch.Tensor]:
        """Retrieve the top-1 matching image for each text query.

        Args:
            prompts (list[str]): List of text queries.

        Returns:
            tuple: A tuple containing:
                - list: Selected frames corresponding to the highest-scoring
                image for each query.
                - Tensor: Indices of the selected frames in ``self.frame_list``.
        """
        text_features = self._get_text_features(prompts, normalize=normalize)
        logits = text_features @ self.image_features.T
        chosen_frame_indices = torch.argmax(logits, dim=-1).flatten()
        chosen_frames = [self.frame_list[i] for i in chosen_frame_indices]
        return chosen_frames, chosen_frame_indices
    
    def query_topk(self, 
                   prompts: List[str], 
                   topk: int, 
                   normalize: bool = False) -> tuple[list[list[PIL.Image]], torch.Tensor]:
        """Retrieve the top-k matching images for each text query.

        Args:
            prompts (list[str]): List of text queries.
            topk (int): Number of top-scoring images to retrieve per query.
            normalize (bool): Whether to L2-normalize the text embeddings.

        Returns:
            tuple: A tuple containing:
                - list[list]: Selected frames corresponding to the top-k
                images for each query.
                - Tensor: Indices of the selected frames in ``self.frame_list``
                for each query.
        """
        text_features = self._get_text_features(prompts, normalize=normalize)
        logits = text_features @ self.image_features.T
        _, chosen_frame_indices = torch.topk(logits, k=topk, dim=-1)
        chosen_frames = [[self.frame_list[i] for i in chosen_frame_indices[j]] \
                         for j in range(chosen_frame_indices.shape[0])]
        return chosen_frames, chosen_frame_indices