"""
GroundingDINO zero-shot pseudo-labeler (Apache 2.0).

IDEA-Research/grounding-dino-tiny  — 172M params, faster
IDEA-Research/grounding-dino-base  — 341M params, more accurate

Text prompt format: "class1. class2. class3."  (period-separated, trailing period)
Labels returned are raw decoded token spans — use COCOBuilder._match_label() to
map them back to canonical class name strings.
"""
import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm


class GroundingDINOLabeler:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_thresh = box_threshold
        self.text_thresh = text_threshold
        print(f"Loading Grounding DINO: {model_id} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        self.model.eval()

    def label_image(self, image_path: str, class_names: List[str]) -> Dict:
        image = Image.open(image_path).convert("RGB")
        text_prompt = ". ".join(class_names) + "."
        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh,
            target_sizes=[image.size[::-1]],
        )[0]
        return {
            "image_path": image_path,
            "boxes_xyxy": results["boxes"].cpu().numpy(),
            "scores": results["scores"].cpu().numpy(),
            "labels": results["labels"],
            "image_width": image.width,
            "image_height": image.height,
        }

    def label_batch(
        self, image_paths: List[str], class_names: List[str]
    ) -> List[Dict]:
        results = []
        for path in tqdm(image_paths, desc="Grounding DINO labeling"):
            try:
                results.append(self.label_image(path, class_names))
            except Exception as e:
                print(f"Error labeling {path}: {e}")
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
