"""
OWLv2 pseudo-labeler — Google's open-vocabulary detector.

Drop-in alternative to GroundingDINOLabeler. Switch via pipeline_config.yaml:
  pseudo_labeling.labeler: "owl_vit"

OWLv2 tends to perform better on small objects and at higher resolutions.
"""
import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm import tqdm


class OWLv2Labeler:
    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        score_threshold: float = 0.20,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = score_threshold
        print(f"Loading OWLv2: {model_id} on {self.device}")
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()

    def label_image(self, image_path: str, class_names: List[str]) -> Dict:
        image = Image.open(image_path).convert("RGB")
        # OWLv2 benefits from descriptive prompts
        texts = [[f"a photo of a {c}" for c in class_names]]
        inputs = self.processor(
            text=texts, images=image, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
        )[0]
        label_ids = results["labels"].cpu().numpy()
        return {
            "image_path": image_path,
            "boxes_xyxy": results["boxes"].cpu().numpy(),
            "scores": results["scores"].cpu().numpy(),
            "labels": [class_names[i] for i in label_ids],
            "image_width": image.width,
            "image_height": image.height,
        }

    def label_batch(
        self, image_paths: List[str], class_names: List[str]
    ) -> List[Dict]:
        results = []
        for path in tqdm(image_paths, desc="OWLv2 labeling"):
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
