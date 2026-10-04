"""
CLIP-based frame pre-filter: cheaply discard frames unlikely to contain the target.

Why CLIP pre-filtering helps
────────────────────────────
Running GroundingDINO or OWLv2 on every extracted frame is slow — a typical
5-minute video at 0.5 fps yields ~150 frames, and the detector takes 0.5–2s
per frame on a GPU. Many frames from wildlife/animal videos are backgrounds,
transition shots, or completely unrelated content.

CLIP (Contrastive Language–Image Pretraining, OpenAI, MIT license) can compute
an image-text similarity score in milliseconds because it only requires a single
forward pass through a ViT-B/32 image encoder and a text transformer. Using CLIP
as a gatekeeper, we can discard ~30-60% of frames before the expensive detector
runs, significantly reducing total pseudo-labeling time.

How CLIP similarity works
─────────────────────────
CLIP maps both images and text into a shared 512-dimensional embedding space.
The similarity between an image and a text prompt is the cosine similarity of
their embeddings. A score of 0.20 is a reasonable threshold: images genuinely
showing the class score 0.25–0.35, while empty backgrounds score ~0.10–0.18.

Multi-prompt averaging
──────────────────────
Using a single prompt like "raccoon" tends to be too strict — CLIP embeds the
bare word differently from photos of raccoons. Averaging three prompt variants:
  "a photo of a raccoon"
  "a clear image of raccoon"
  "raccoon"
gives a more robust, recall-friendly text embedding that better matches how
real photos of raccoons look in CLIP's image embedding space.

License: openai/clip-vit-base-patch32 — MIT (both the model and the code)
"""
import torch
from typing import List, Tuple, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


class CLIPFilter:
    """
    Filters a list of image frames by their CLIP similarity to a class name.

    Frames below the similarity threshold are discarded before the more
    expensive pseudo-labeler (GroundingDINO or OWLv2) runs on the remainder.

    This is a recall-side filter: it is intentionally lenient (low default
    threshold 0.20) to avoid discarding true positives at the expense of some
    false positives that the labeler's own confidence threshold will handle.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        similarity_threshold: float = 0.20,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name:           HuggingFace model ID for the CLIP checkpoint.
                                  clip-vit-base-patch32 is small and fast.
                                  clip-vit-large-patch14 is more accurate but slower.
            similarity_threshold: Minimum cosine similarity to keep a frame.
                                  Lower → fewer frames discarded (higher recall).
                                  Higher → more frames discarded (faster labeling).
                                  Tune based on clip_filtering.similarity_threshold
                                  in pipeline_config.yaml.
            batch_size:           How many images to encode per CLIP forward pass.
                                  32 is safe for 4GB GPU; increase for larger GPUs.
            device:               "cuda", "cpu", or None (auto-detect).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = similarity_threshold
        self.batch_size = batch_size
        print(f"Loading CLIP: {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # inference only; dropout disabled

    def _encode_text(self, class_name: str) -> torch.Tensor:
        """
        Compute a text embedding for a class name using prompt ensembling.

        Returns a single normalized (1, D) embedding that is the L2-normalized
        mean of embeddings from three prompt templates. This technique, described
        in the original CLIP paper, improves zero-shot accuracy by ~3–5% vs
        using a bare class name, because it better captures how the class looks
        in natural photographs.
        """
        prompts = [
            f"a photo of a {class_name}",   # most common prompt template
            f"a clear image of {class_name}",  # emphasizes visual clarity
            f"{class_name}",                 # bare word as anchor
        ]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            # L2-normalize so cosine similarity = dot product (numerically stable)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            # Average across the three prompts
            feats = feats.mean(dim=0, keepdim=True)
            # Re-normalize after averaging so the final embedding is unit length
            return feats / feats.norm(dim=-1, keepdim=True)

    def filter_frames(
        self,
        image_paths: List[str],
        class_name: str,
    ) -> Tuple[List[str], List[float]]:
        """
        Filter image paths by CLIP similarity to a class name.

        Processing strategy:
          - Text embedding is computed once and reused for all batches.
          - Images are processed in batches of self.batch_size to use GPU
            tensor cores efficiently (batched matrix multiply in CLIP's ViT).
          - Images that fail to load are silently skipped (corrupt/missing files
            should not abort the whole filter run).

        Args:
            image_paths: List of paths to JPEG/PNG frames.
            class_name:  Target class name string (e.g. "raccoon").

        Returns:
            (kept_paths, similarity_scores):
              kept_paths         : paths of frames that passed the threshold
              similarity_scores  : cosine similarity score for each kept frame
        """
        # Compute text embedding once — reused across all image batches
        text_feats = self._encode_text(class_name)
        kept: List[str] = []
        scores: List[float] = []

        for i in tqdm(
            range(0, len(image_paths), self.batch_size),
            desc=f"CLIP filter [{class_name}]",
        ):
            batch = image_paths[i : i + self.batch_size]

            # Load all images in this batch; skip files that fail to open
            imgs, valid = [], []
            for p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    valid.append(p)
                except Exception:
                    pass  # silently skip unreadable frames
            if not imgs:
                continue

            # Encode the batch of images into CLIP's embedding space
            inputs = self.processor(
                images=imgs, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                img_feats = self.model.get_image_features(**inputs)
                # L2-normalize image features before computing cosine similarity
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                # Batch matrix multiply: (batch, D) @ (D, 1) → (batch, 1)
                # squeeze(1) → (batch,) cosine similarity scores
                sims = (img_feats @ text_feats.T).squeeze(1).cpu().numpy()

            for path, score in zip(valid, sims):
                if score >= self.threshold:
                    kept.append(path)
                    scores.append(float(score))

        print(f"CLIP: kept {len(kept)}/{len(image_paths)} frames")
        return kept, scores
