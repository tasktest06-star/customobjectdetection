"""
Grounding DINO pseudo-labeler — zero-shot open-vocabulary object detection.

What Grounding DINO does
────────────────────────
Grounding DINO fuses a DINO vision transformer backbone with a BERT-style text
encoder using cross-modal attention. At inference time it accepts:
  - An image
  - A text prompt listing class names separated by periods

It outputs bounding boxes whose features are matched to the text spans via
a learned cross-modal alignment. No task-specific training is required — the
model was trained on large grounded image-text datasets (Objects365, GoldG,
Cap4M) and generalizes to arbitrary class names at runtime.

Text prompt format
──────────────────
GroundingDINO expects class names joined with ". " and a trailing period:
  "raccoon. capybara."          ← correct
  "raccoon, capybara"           ← wrong (comma separator doesn't work well)
  "a raccoon. a capybara."      ← also works (articles help recall)

The model's text decoder outputs label spans as raw decoded text rather than
integer class IDs. For example, for class "raccoon" it might return "raccoon."
or "racoon" depending on tokenization. SelfTrainer._match_label() handles
this normalization during the teacher-student ensemble step.

License: Apache 2.0 (IDEA-Research/grounding-dino-tiny on HuggingFace)
"""
import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm


class GroundingDINOLabeler:
    """
    Zero-shot object detector that maps free-text class names to bounding boxes.

    Two checkpoints available (both Apache 2.0):
      "IDEA-Research/grounding-dino-tiny"  — ~172M params, faster, less accurate
      "IDEA-Research/grounding-dino-base"  — ~341M params, slower, more accurate

    Set via pseudo_labeling.model in pipeline_config.yaml.

    Thresholds:
      box_threshold:  Minimum detection confidence score. Lower → more detections
                      (higher recall, higher noise). Typical range: 0.20–0.40.
      text_threshold: Minimum text-image alignment score. Controls whether a box
                      is matched to a specific class. Typically slightly lower
                      than box_threshold.

    Setting box_threshold too low: many false positives in backgrounds.
    Setting it too high: misses occluded or small instances.
    Start at 0.30 and lower to 0.20 if too few pseudo-labels are generated.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_id:        HuggingFace model ID for the GroundingDINO checkpoint.
            box_threshold:   Minimum score to keep a bounding box prediction.
            text_threshold:  Minimum score for text-to-box alignment matching.
            device:          "cuda", "cpu", or None (auto-detect).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_thresh = box_threshold
        self.text_thresh = text_threshold
        print(f"Loading Grounding DINO: {model_id} on {self.device}")
        # AutoProcessor handles tokenization + image preprocessing for GroundingDINO
        self.processor = AutoProcessor.from_pretrained(model_id)
        # AutoModelForZeroShotObjectDetection dispatches to the right architecture
        # for zero-shot detection (GroundingDINO uses a different arch than OWLv2)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        self.model.eval()  # disable dropout; we never train this model

    def label_image(self, image_path: str, class_names: List[str]) -> Dict:
        """
        Detect objects in one image using zero-shot text prompting.

        Text prompt format: GroundingDINO requires class names joined with ". "
        with a trailing period. This format matches the separator used during
        the model's pre-training on grounded image-caption data.

        Returns a dict with:
          "image_path"   : str
          "boxes_xyxy"   : np.ndarray (N, 4) — pixel coords, [x1, y1, x2, y2]
          "scores"       : np.ndarray (N,)   — confidence scores in [0, 1]
          "labels"       : List[str]         — decoded label spans (may include
                                               trailing periods or partial tokens)
          "image_width"  : int
          "image_height" : int

        Note: "labels" contains raw decoded text tokens, not clean class names.
        Use SelfTrainer._match_label() or COCOBuilder._match_label() to map
        these back to your canonical class name strings before using them.
        """
        image = Image.open(image_path).convert("RGB")

        # Build the period-separated text prompt required by GroundingDINO.
        # Example: class_names=["raccoon", "capybara"] → "raccoon. capybara."
        # The trailing period is required — without it the last class may be missed.
        text_prompt = ". ".join(class_names) + "."

        # Processor handles both the image (resize/normalize) and text (tokenize)
        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # post_process_grounded_object_detection rescales predicted boxes from
        # normalized [0,1] coordinates back to pixel coordinates using target_sizes.
        # target_sizes format: list of (height, width) tuples — note height-first.
        # image.size returns (width, height) so we reverse with [::-1].
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,        # needed to decode text label spans
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh,
            target_sizes=[image.size[::-1]],  # [(height, width)]
        )[0]  # [0] because we passed one image; processor returns list of per-image dicts

        return {
            "image_path":   image_path,
            "boxes_xyxy":   results["boxes"].cpu().numpy(),
            "scores":       results["scores"].cpu().numpy(),
            "labels":       results["labels"],   # raw decoded token strings
            "image_width":  image.width,
            "image_height": image.height,
        }

    def label_batch(
        self, image_paths: List[str], class_names: List[str]
    ) -> List[Dict]:
        """
        Label a list of images.

        Processes images one at a time (no batching) because GroundingDINO's
        cross-modal attention requires a text prompt per image and the processor
        does not support efficient batching of image+text pairs with variable
        box counts.

        Errors on individual images are caught and returned as empty detections
        so a single bad frame (corrupted JPEG, permission error) doesn't abort
        the entire labeling run. These empty dets are filtered out by
        LabelQualityFilter.filter_batch().
        """
        results = []
        for path in tqdm(image_paths, desc="Grounding DINO labeling"):
            try:
                results.append(self.label_image(path, class_names))
            except Exception as e:
                print(f"Error labeling {path}: {e}")
                # Return an empty detection dict with the correct schema so
                # downstream code doesn't need to handle missing keys
                results.append(
                    {
                        "image_path":   path,
                        "boxes_xyxy":   np.zeros((0, 4)),
                        "scores":       np.array([]),
                        "labels":       [],
                        "image_width":  0,
                        "image_height": 0,
                    }
                )
        return results
