"""
Pipeline step checkpointing.

Saves output of each pipeline step to disk so a crash at step N
resumes from step N-1 instead of restarting from scratch.
State metadata is JSON; large objects (frame lists, detections) are pickle.
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class PipelineCheckpoint:
    """
    Usage:
        ckpt = PipelineCheckpoint(run_name="my_run")

        if ckpt.is_done("extract"):
            frames = ckpt.load("extract")
        else:
            frames = extractor.extract_all(...)
            ckpt.save("extract", frames)
    """

    STEPS: List[str] = [
        "download",
        "extract",
        "clip_filter",
        "pseudo_label",
        "quality_filter",
        "build_dataset",
        "train",
    ]

    def __init__(
        self,
        checkpoint_dir: str = ".pipeline_checkpoints",
        run_name: str = "run",
    ):
        self.dir = Path(checkpoint_dir) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.dir / "meta.json"
        self._meta = self._load_meta()

    # ── persistence ────────────────────────────────────────────────

    def _load_meta(self) -> Dict:
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                return json.load(f)
        return {"completed_steps": {}}

    def _flush(self):
        with open(self._meta_path, "w") as f:
            json.dump(self._meta, f, indent=2, default=str)

    # ── public API ─────────────────────────────────────────────────

    def is_done(self, step: str) -> bool:
        """True if this step completed in a previous run."""
        return step in self._meta["completed_steps"]

    def save(self, step: str, data: Any) -> None:
        """Persist step output to pickle and record metadata."""
        pkl = self.dir / f"{step}.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._meta["completed_steps"][step] = {
            "timestamp": datetime.now().isoformat(),
            "pkl": str(pkl),
        }
        self._flush()

    def load(self, step: str) -> Any:
        """Load a previously saved step output."""
        info = self._meta["completed_steps"].get(step)
        if info is None:
            raise KeyError(f"No checkpoint found for step '{step}'")
        with open(info["pkl"], "rb") as f:
            return pickle.load(f)

    def reset(self, step: Optional[str] = None) -> None:
        """
        Clear checkpoint for one step (or all steps if step is None).
        Also removes the pickle file.
        """
        if step is None:
            for s, info in self._meta["completed_steps"].items():
                _try_remove(info.get("pkl"))
            self._meta["completed_steps"] = {}
        elif step in self._meta["completed_steps"]:
            _try_remove(self._meta["completed_steps"][step].get("pkl"))
            del self._meta["completed_steps"][step]
        self._flush()

    def status(self) -> Dict[str, str]:
        """Return a human-readable status dict for all steps."""
        return {
            s: ("✅ done  @ " + self._meta["completed_steps"][s]["timestamp"]
                if self.is_done(s) else "⬜ pending")
            for s in self.STEPS
        }

    def print_status(self) -> None:
        print("\n── Checkpoint status ───────────────────────")
        for step, state in self.status().items():
            print(f"  {step:<20} {state}")
        print("────────────────────────────────────────────\n")


def _try_remove(path: Optional[str]) -> None:
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
