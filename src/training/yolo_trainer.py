"""YOLOv8 fine-tuning, evaluation, and inference wrapper."""
from pathlib import Path
from typing import Dict
from ultralytics import YOLO


class YOLOTrainer:
    def __init__(
        self,
        base_model: str = "yolov8n.pt",
        output_dir: str = "models",
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        run_name: str = "run",
        **kwargs,
    ) -> str:
        """Fine-tune on a custom dataset. Returns path to best.pt weights."""
        model = YOLO(self.base_model)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=str(self.output_dir),
            name=run_name,
            patience=10,
            save=True,
            plots=True,
            **kwargs,
        )
        best = Path(results.save_dir) / "weights" / "best.pt"
        print(f"Best model saved to: {best}")
        return str(best)

    def evaluate(self, model_path: str, data_yaml: str) -> Dict[str, float]:
        """Run validation and return standard detection metrics."""
        model = YOLO(model_path)
        m = model.val(data=data_yaml)
        return {
            "mAP50": float(m.box.map50),
            "mAP50-95": float(m.box.map),
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
        }

    def predict(self, model_path: str, source: str, conf: float = 0.25):
        """Run inference. source can be image path, directory, or video."""
        model = YOLO(model_path)
        return model.predict(source=source, conf=conf, save=True)
