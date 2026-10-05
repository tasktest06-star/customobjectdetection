"""
CLIP-based frame pre-filter: discard frames unlikely to contain the target class.

Uses prompt ensembling (3 templates averaged) for more robust text embeddings.
Model: openai/clip-vit-base-patch32 (MIT license — code and weights).
"""
import torch
from typing import List, Tuple, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


class CLIPFilter:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        similarity_threshold: float = 0.20,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = similarity_threshold
        self.batch_size = batch_size
        print(f"Loading CLIP: {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def _encode_text(self, class_name: str) -> torch.Tensor:
        prompts = [
            f"a photo of a {class_name}",
            f"a clear image of {class_name}",
            f"{class_name}",
        ]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.mean(dim=0, keepdim=True)
            return feats / feats.norm(dim=-1, keepdim=True)

    def filter_frames(
        self, image_paths: List[str], class_name: str
    ) -> Tuple[List[str], List[float]]:
        text_feats = self._encode_text(class_name)
        kept: List[str] = []
        scores: List[float] = []
        for i in tqdm(
            range(0, len(image_paths), self.batch_size),
            desc=f"CLIP filter [{class_name}]",
        ):
            batch = image_paths[i : i + self.batch_size]
            imgs, valid = [], []
            for p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    valid.append(p)
                except Exception:
                    pass
            if not imgs:
                continue
            inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                img_feats = self.model.get_image_features(**inputs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                sims = (img_feats @ text_feats.T).squeeze(1).cpu().numpy()
            for path, score in zip(valid, sims):
                if score >= self.threshold:
                    kept.append(path)
                    scores.append(float(score))
        print(f"CLIP: kept {len(kept)}/{len(image_paths)} frames")
        return kept, scores
