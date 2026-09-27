from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as TVF

from src.data.datasets.constants import CAMERA_IMAGE_SIZE, INPUT_IMAGE_SIZE
from src.utils.camera_functions import image_affine

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKED_DIR = REPO_ROOT / "data" / "perception"
INPUT_AFFINE = image_affine(CAMERA_IMAGE_SIZE, INPUT_IMAGE_SIZE)

# Forked workers share these cached dictionaries through copy-on-write.
_PACKED_CACHE: dict[Path, dict] = {}


def load_packed(name: str) -> dict:
    """Load a cached modality dictionary: {sequence: {frame_index: data}}."""
    path = (PACKED_DIR / f"{name}.pkl").resolve()
    if path not in _PACKED_CACHE:
        if not path.is_file():
            raise FileNotFoundError(
                f"Packed modality file not found: {path}. Run `bash scripts/download.sh` to download it."
            )
        with path.open("rb") as fh:
            _PACKED_CACHE[path] = pickle.load(fh)
    return _PACKED_CACHE[path]


class ModalityIOMixin:
    """Per-frame modality loading helpers for ``MMORDataset``."""

    def _load_pose_data(self, frame_meta: dict) -> list[dict]:
        labeled_poses = self._pose_packed.get(frame_meta["sequence"], {}).get(frame_meta["pose_key"], {})
        return [
            {"label": label, "keypoints": np.asarray(kps, dtype=np.float32).copy()}  # (J, 4) = xyz + confidence
            for label, kps in labeled_poses.items()
            if label != "unknown"
        ]

    def _load_screen_txt_summary(self, frame_meta: dict) -> torch.Tensor:
        t = self._txt_packed.get(frame_meta["sequence"], {}).get(frame_meta["txt_key"])
        return t.clone() if t is not None else torch.zeros((768,), dtype=torch.float32)

    def _load_screen_json_summary(self, frame_meta: dict) -> torch.Tensor:
        t = self._json_packed.get(frame_meta["sequence"], {}).get(frame_meta["json_key"])
        return t.clone() if t is not None else torch.zeros((768,), dtype=torch.float32)

    def _load_audio_embedding(self, frame_meta: dict) -> torch.Tensor:
        t = self._audio_packed.get(frame_meta["sequence"], {}).get(frame_meta["audio_key"])
        return t.clone() if t is not None else torch.zeros((512,), dtype=torch.float32)

    def _load_cv2_image(self, path: Path, resize, fallback_shape) -> torch.Tensor:
        """Decode, resize and ImageNet-normalize an image; missing or unreadable files are black."""
        img = None
        if path.is_file():
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if bgr is not None:
                img = resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img_tensor = (
            torch.zeros(fallback_shape, dtype=torch.uint8)
            if img is None
            else torch.from_numpy(img).permute(2, 0, 1)
        )
        return TVF.normalize(img_tensor.float() / 255.0, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _load_screen_image(self, frame_meta: dict) -> torch.Tensor:
        image = self._load_cv2_image(
            frame_meta["screen_img_path"],
            lambda img: cv2.resize(img, (128, 72), interpolation=cv2.INTER_AREA),
            (3, 72, 128),
        )
        return image.unsqueeze(0)

    def _load_color_images(self, frame_meta: dict) -> torch.Tensor:
        images = [
            self._load_cv2_image(
                color_path,
                lambda img: cv2.warpAffine(img, INPUT_AFFINE, INPUT_IMAGE_SIZE, flags=cv2.INTER_LINEAR),
                (3, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]),
            )
            for color_path in frame_meta["colorimage_paths"]
        ]
        return torch.stack(images).unsqueeze(0)
