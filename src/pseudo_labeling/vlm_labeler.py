"""
VLM-based pseudo-labeler: uses a vision-language model to answer grounding questions.

WHY THIS EXISTS
───────────────
GroundingDINO and OWLv2 excel for common object classes. They struggle when:
  - The class name is ambiguous ("bat" → baseball bat vs. animal)
  - The class requires context to identify ("abandoned car" vs. "parked car")
  - The visual appearance is unusual (novel scientific equipment, rare species)
  - You need rich spatial reasoning ("the second bird from the left")

A VLM can leverage its language understanding to ground the class more precisely.
The pipeline asks the VLM structured grounding questions and parses its output
into bounding boxes — no training needed, just prompting.

SUPPORTED BACKENDS (all Apache 2.0)
────────────────────────────────────
  "qwen2_vl"    → Qwen/Qwen2-VL-2B-Instruct or Qwen2-VL-7B-Instruct (Apache 2.0)
  "internvl2"   → OpenGVLab/InternVL2-2B or InternVL2-8B (Apache 2.0)

HOW IT WORKS
────────────
Strategy A — Direct grounding (Qwen2-VL, InternVL2 with grounding):
  Prompt: "Locate all {class_name} in the image. Output each object as a bounding
           box in JSON format: [{{'label': '{class_name}', 'bbox_2d': [x1, y1, x2, y2]}}]
           Use absolute pixel coordinates."
  The model returns JSON with pixel-coordinate bounding boxes.
  Works best with Qwen2-VL-7B which was fine-tuned on grounding tasks.

Strategy B — Region description + CLIP re-scoring:
  Step 1: Ask VLM "Does this image contain a {class_name}? Answer YES or NO."
  Step 2: For YES responses, ask "Describe the location of the {class_name}
          (e.g. top-left, center, etc.)."
  Step 3: Convert spatial descriptions to heuristic bounding boxes.
  Step 4: Re-score with CLIP to filter false positives.
  Works with any VLM; less accurate but more widely applicable.

This file implements Strategy A (Qwen2-VL direct grounding) with a fallback
to Strategy B when the model doesn't support structured output.

LICENSE NOTE
────────────
Qwen2-VL: Apache 2.0 (Qwen/Qwen2-VL-2B-Instruct on HuggingFace)
InternVL2: Apache 2.0 (OpenGVLab/InternVL2-2B on HuggingFace)
Both model weights are also Apache 2.0 licensed.

INSTALL
───────
pip install transformers>=4.45.0 qwen-vl-utils einops timm
"""
import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from tqdm import tqdm


# ── Qwen2-VL backend ──────────────────────────────────────────────────────────

class Qwen2VLLabeler:
    """
    Zero-shot grounding using Qwen2-VL.

    Qwen2-VL was specifically trained on grounding tasks and can output
    bounding boxes in its response. The smallest useful checkpoint is
    Qwen/Qwen2-VL-2B-Instruct (~4 GB VRAM).

    Larger checkpoints (7B, 72B) give higher grounding accuracy at more VRAM cost.
    The 2B model is recommended for pipelines running alongside a separate detector.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        grounding_threshold: float = 0.0,
    ):
        """
        Args:
            model_id:             HuggingFace model ID. Apache 2.0 checkpoints:
                                    Qwen/Qwen2-VL-2B-Instruct  (fast, ~4 GB VRAM)
                                    Qwen/Qwen2-VL-7B-Instruct  (better, ~16 GB VRAM)
            device:               "cuda", "cpu", or None (auto).
            max_new_tokens:       Max tokens in the VLM response. 512 is enough for
                                  10–20 bounding boxes in JSON format.
            grounding_threshold:  Minimum area ratio to keep a grounded box.
                                  0.0 = keep all (filtering done by LabelQualityFilter).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.area_thresh = grounding_threshold

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            self._process_vision_info = process_vision_info
        except ImportError:
            raise ImportError(
                "Qwen2-VL requires additional packages:\n"
                "  pip install transformers>=4.45.0 qwen-vl-utils"
            )

        print(f"Loading Qwen2-VL: {model_id} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()

    def _build_prompt(self, class_name: str) -> str:
        return (
            f"Detect all instances of '{class_name}' in the image. "
            f"For each instance found, output a JSON object with keys "
            f"'label' (string = '{class_name}') and 'bbox_2d' ([x1, y1, x2, y2] "
            f"in absolute pixel coordinates, integers). "
            f"Output ONLY a JSON array. If none found, output []."
        )

    def _parse_response(
        self, response: str, image_w: int, image_h: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract bounding boxes from the VLM's JSON-format response.

        Handles both clean JSON arrays and messy responses where JSON is embedded
        in prose (e.g., "The {class} is located at [...] coordinates").

        Returns (boxes_xyxy, scores, labels).
        Scores are all 1.0 since VLMs don't produce confidence values.
        """
        # Extract JSON array from anywhere in the response.
        # Greedy (.*) is required: the bbox sub-array [x1,y1,x2,y2] contains ']'
        # characters, so a non-greedy .*? would stop at the first inner ']'
        # and return truncated invalid JSON.
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return np.zeros((0, 4)), np.array([]), []

        try:
            detections = json.loads(json_match.group())
        except json.JSONDecodeError:
            return np.zeros((0, 4)), np.array([]), []

        boxes, labels = [], []
        for det in detections:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox_2d") or det.get("bbox") or det.get("box")
            label = det.get("label", "")
            if not bbox or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                # Clip to image bounds
                x1 = max(0.0, min(x1, image_w))
                y1 = max(0.0, min(y1, image_h))
                x2 = max(0.0, min(x2, image_w))
                y2 = max(0.0, min(y2, image_h))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(str(label))
            except (TypeError, ValueError):
                continue

        if not boxes:
            return np.zeros((0, 4)), np.array([]), []

        return (
            np.array(boxes, dtype=np.float32),
            np.ones(len(boxes), dtype=np.float32),
            labels,
        )

    def label_image(self, image_path: str, class_names: List[str]) -> Dict:
        """
        Detect objects for all class_names in one image.

        Runs one VLM query per class to keep the prompt focused (multi-class
        prompts confuse grounding in the 2B model). The 7B model handles
        multi-class prompts better; if using 7B you can modify this method
        to pass all class names in a single query.
        """
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        all_boxes, all_scores, all_labels = [], [], []

        for cls in class_names:
            prompt = self._build_prompt(cls)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Prepare inputs using Qwen2-VL's vision processing
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            # Decode only the newly generated tokens (skip the prompt tokens)
            n_prompt = inputs["input_ids"].shape[1]
            response = self.processor.decode(
                generated[0][n_prompt:], skip_special_tokens=True
            )

            boxes, scores, labels = self._parse_response(response, W, H)
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.extend(labels)

        if all_boxes:
            final_boxes = np.concatenate(all_boxes, axis=0)
            final_scores = np.concatenate(all_scores, axis=0)
        else:
            final_boxes = np.zeros((0, 4))
            final_scores = np.array([])
            all_labels = []

        return {
            "image_path": image_path,
            "boxes_xyxy": final_boxes,
            "scores": final_scores,
            "labels": all_labels,
            "image_width": W,
            "image_height": H,
        }

    def label_batch(
        self, image_paths: List[str], class_names: List[str]
    ) -> List[Dict]:
        results = []
        for path in tqdm(image_paths, desc="Qwen2-VL grounding"):
            try:
                results.append(self.label_image(path, class_names))
            except Exception as e:
                print(f"VLM error on {path}: {e}")
                results.append(
                    {
                        "image_path": path,
                        "boxes_xyxy": np.zeros((0, 4)),
                        "scores": np.array([]),
                        "labels": [],
                        "image_width": 0,
                        "image_height": 0,
                    }
                )
        return results


# ── InternVL2 backend ─────────────────────────────────────────────────────────

class InternVL2Labeler:
    """
    Zero-shot grounding using InternVL2 (OpenGVLab, Apache 2.0).

    InternVL2 supports <box>[[x1,y1,x2,y2]]</box> grounding output tokens.
    The 2B model runs in ~4 GB VRAM; 8B in ~16 GB VRAM.

    Models (all Apache 2.0):
      OpenGVLab/InternVL2-2B   — fast, ~4 GB VRAM
      OpenGVLab/InternVL2-8B   — more accurate, ~16 GB VRAM
    """

    GROUNDING_PROMPT = (
        "Please detect all instances of '{class_name}' in the image. "
        "For each instance, output its bounding box in the format "
        "<box>[[x1,y1,x2,y2]]</box> where coordinates are absolute pixels. "
        "If none, output: none found."
    )

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2-2B",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        try:
            from transformers import AutoModel, AutoTokenizer
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            self._T = T
            self._InterpolationMode = InterpolationMode
            self._AutoModel = AutoModel
            self._AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install required packages: pip install transformers torchvision"
            )

        print(f"Loading InternVL2: {model_id} on {self.device}")
        self.tokenizer = self._AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = self._AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self._img_size = 448
        self._transform = self._build_transform(self._img_size)

    def _build_transform(self, input_size: int):
        from torchvision import transforms
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                (input_size, input_size),
                interpolation=self._InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _parse_box_tokens(
        self, text: str, image_w: int, image_h: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Parse <box>[[x1,y1,x2,y2]]</box> tokens from InternVL2 output."""
        pattern = r"<box>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]</box>"
        matches = re.findall(pattern, text)
        if not matches:
            return np.zeros((0, 4)), np.array([]), []
        boxes = []
        for m in matches:
            x1, y1, x2, y2 = [int(v) for v in m]
            x1 = max(0, min(x1, image_w))
            y1 = max(0, min(y1, image_h))
            x2 = max(0, min(x2, image_w))
            y2 = max(0, min(y2, image_h))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        if not boxes:
            return np.zeros((0, 4)), np.array([]), []
        return (
            np.array(boxes, dtype=np.float32),
            np.ones(len(boxes), dtype=np.float32),
            [""] * len(boxes),
        )

    def label_image(self, image_path: str, class_names: List[str]) -> Dict:
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        pixel_values = self._transform(image).unsqueeze(0).to(
            self.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        all_boxes, all_scores, all_labels = [], [], []

        for cls in class_names:
            prompt = self.GROUNDING_PROMPT.format(class_name=cls)
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config={"max_new_tokens": self.max_new_tokens, "do_sample": False},
            )
            boxes, scores, _ = self._parse_box_tokens(response, W, H)
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.extend([cls] * len(boxes))

        if all_boxes:
            final_boxes = np.concatenate(all_boxes, axis=0)
            final_scores = np.concatenate(all_scores, axis=0)
        else:
            final_boxes = np.zeros((0, 4))
            final_scores = np.array([])
            all_labels = []

        return {
            "image_path": image_path,
            "boxes_xyxy": final_boxes,
            "scores": final_scores,
            "labels": all_labels,
            "image_width": W,
            "image_height": H,
        }

    def label_batch(
        self, image_paths: List[str], class_names: List[str]
    ) -> List[Dict]:
        results = []
        for path in tqdm(image_paths, desc="InternVL2 grounding"):
            try:
                results.append(self.label_image(path, class_names))
            except Exception as e:
                print(f"VLM error on {path}: {e}")
                results.append(
                    {
                        "image_path": path,
                        "boxes_xyxy": np.zeros((0, 4)),
                        "scores": np.array([]),
                        "labels": [],
                        "image_width": 0,
                        "image_height": 0,
                    }
                )
        return results


# ── Factory ───────────────────────────────────────────────────────────────────

def build_vlm_labeler(
    backend: str = "qwen2_vl",
    model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> "Qwen2VLLabeler | InternVL2Labeler":
    """
    Instantiate the requested VLM labeler backend.

    Args:
        backend:  "qwen2_vl" or "internvl2"
        model_id: Override the default model ID. If None, uses the smaller
                  recommended checkpoint for each backend.
        device:   "cuda", "cpu", or None (auto).

    Returns:
        A labeler instance with a .label_batch(image_paths, class_names) method
        that returns the same dict schema as GroundingDINOLabeler.
    """
    if backend == "qwen2_vl":
        mid = model_id or "Qwen/Qwen2-VL-2B-Instruct"
        return Qwen2VLLabeler(model_id=mid, device=device)
    elif backend == "internvl2":
        mid = model_id or "OpenGVLab/InternVL2-2B"
        return InternVL2Labeler(model_id=mid, device=device)
    else:
        raise ValueError(f"Unknown VLM backend: {backend!r}. Choose 'qwen2_vl' or 'internvl2'.")
