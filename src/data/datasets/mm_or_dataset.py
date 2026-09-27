import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from torch_geometric.data import Data

from src.data.graph_transforms import GraphAugmentor
from src.data.temporal_graph import TemporalSuperGraph, build_causal_super_graph
from src.utils.camera_functions import get_cam_params

from .constants import CAMERAS, OR_CENTER
from .geometry import _jitter_world_coords
from .index import IndexBuildMixin, IndexCacheMixin
from .modality_io import INPUT_AFFINE, ModalityIOMixin, load_packed
from .scene_graph import build_scene_graph

_OR_CENTER = OR_CENTER.astype(np.float64)


class MMORDataset(IndexBuildMixin, IndexCacheMixin, ModalityIOMixin, data.Dataset):
    """Windows of ``temporal_window`` consecutive MM-OR frames, each one scene graph.

    Training windows start every ``train_stride`` frames and draw coordinate jitter,
    rigid world augmentation and graph augmentation; test windows start at every frame.
    Rigid augmentation rotates and translates positions about the OR centre; box
    orientations and camera extrinsics are left unchanged.
    """

    def __init__(
        self,
        split,
        data_dir,
        sequences,
        temporal_window=8,
        train_stride=4,
        test_stride=1,
        augmentation=None,
        coord_noise_std_mm=8.0,
        rigid_aug_rotation_deg=0.0,
        rigid_aug_translation_mm=0.0,
        max_frames_per_seq=None,
        use_index_cache=True,
    ):
        train = split == "train"
        self.split = split
        self.log = logging.getLogger(__name__)
        self.data_root = Path(data_dir)
        self.sequences = sequences
        self.temporal_window = temporal_window
        self.stride = train_stride if train else test_stride
        self.max_frames_per_seq = max_frames_per_seq
        self.augmentor = GraphAugmentor(**augmentation) if train and augmentation is not None else None
        self.coord_noise_std_mm = coord_noise_std_mm if train else 0.0
        self.rigid_aug_rotation_deg = rigid_aug_rotation_deg if train else 0.0
        self.rigid_aug_translation_mm = rigid_aug_translation_mm if train else 0.0
        self.cam_params_per_sequence = get_cam_params(sequences, self.data_root, CAMERAS)
        self._bbox_packed = load_packed("bounding_boxes")
        self._pose_packed = load_packed("human_poses")
        self._audio_packed = load_packed("audio_embeddings")
        self._txt_packed = load_packed("text_embeddings")
        self._json_packed = load_packed("json_embeddings")
        if self.coord_noise_std_mm > 0:
            self.log.info("[MMORDataset] Coordinate jitter: noise_std=%.1fmm", self.coord_noise_std_mm)
        self._build_index(use_index_cache)

    def _sample_rigid(self):
        """Sample a world-to-augmented-world transform shared by the temporal window."""
        if self.rigid_aug_rotation_deg <= 0 and self.rigid_aug_translation_mm <= 0:
            return None
        deg = (np.random.rand() * 2 - 1) * self.rigid_aug_rotation_deg
        rad = np.deg2rad(deg)
        c, s = (np.cos(rad), np.sin(rad))
        R_augmented_world = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        t = (np.random.rand(3) * 2 - 1) * self.rigid_aug_translation_mm
        t[2] = 0.0
        T_augmented_world = np.eye(4, dtype=np.float64)
        T_augmented_world[:3, :3] = R_augmented_world
        T_augmented_world[:3, 3] = _OR_CENTER + t - R_augmented_world @ _OR_CENTER
        return T_augmented_world

    def _transform_world(self, coords: np.ndarray, T_augmented_world) -> np.ndarray:
        """Rotate and translate world coordinates about the OR centre, then apply measurement noise."""
        if T_augmented_world is not None:
            rotation = T_augmented_world[:3, :3]
            translation = T_augmented_world[:3, 3] - _OR_CENTER + rotation @ _OR_CENTER
            coords = (coords - _OR_CENTER) @ rotation.T + _OR_CENTER + translation
        if self.coord_noise_std_mm > 0:
            coords = _jitter_world_coords(coords, self.coord_noise_std_mm)
        return coords

    def _jitter_bbox_data(self, bbox_data: dict, T_augmented_world=None) -> dict:
        """Transform box centres; box orientations are left unchanged."""
        if T_augmented_world is None and self.coord_noise_std_mm <= 0:
            return bbox_data
        return {
            name: {
                **box,
                "center": self._transform_world(
                    np.asarray(box["center"], dtype=np.float64).reshape(1, 3), T_augmented_world
                ).reshape(-1),
            }
            for name, box in bbox_data.items()
        }

    def _jitter_pose_data(self, pose_data: list[dict], T_augmented_world=None) -> list[dict]:
        """Transform world-space keypoints without changing confidence values."""
        if T_augmented_world is None and self.coord_noise_std_mm <= 0:
            return pose_data
        out = []
        for pose_info in pose_data:
            new_info = dict(pose_info)
            kps = np.array(pose_info["keypoints"], dtype=np.float32)
            kps[:, :3] = self._transform_world(kps[:, :3].astype(np.float64), T_augmented_world).astype(np.float32)
            new_info["keypoints"] = kps
            out.append(new_info)
        return out

    def _load_frame(self, frame_meta: dict, T_augmented_world=None) -> Data:
        sequence = frame_meta["sequence"]
        bbox_data = self._bbox_packed.get(sequence, {}).get(frame_meta["bbox_key"], {})
        bbox_data = self._jitter_bbox_data(bbox_data, T_augmented_world=T_augmented_world)
        pose_data = self._jitter_pose_data(self._load_pose_data(frame_meta), T_augmented_world=T_augmented_world)
        graph = build_scene_graph(bbox_data, pose_data, frame_meta)
        graph.screen_txt = self._load_screen_txt_summary(frame_meta).unsqueeze(0)
        graph.screen_json = self._load_screen_json_summary(frame_meta).unsqueeze(0)
        graph.audio = self._load_audio_embedding(frame_meta).unsqueeze(0)
        graph.screen_image = self._load_screen_image(frame_meta)
        graph.images = self._load_color_images(frame_meta)
        graph.affine = torch.from_numpy(INPUT_AFFINE).float().unsqueeze(0)
        graph.cam_params = torch.from_numpy(self.cam_params_per_sequence[sequence]).float().unsqueeze(0)
        return graph

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> TemporalSuperGraph:
        return self._get_item(idx)

    def __getitems__(self, indices):
        """Reuse overlapping evaluation frames within one DataLoader batch."""
        if self.split == "train":
            return [self[index] for index in indices]
        frames = {}
        return [self._get_item(index, frames) for index in indices]

    def _get_item(self, idx, frames=None):
        sequence, start_idx = self.samples[idx]
        num_frames = len(self.sequence_frames[sequence])
        window_data = self._load_window(sequence, start_idx, frames)
        graph = build_causal_super_graph(window_data)
        if self.augmentor is not None:
            graph = self.augmentor(graph)
        self._validate_graph(graph)
        last_idx = min(max(start_idx + self.temporal_window - 1, 0), num_frames - 1)
        graph.take_pos = torch.tensor([float(last_idx)], dtype=torch.float32)
        return graph

    def _load_window(self, sequence, start_idx, frames):
        frames_meta = self.sequence_frames[sequence]
        num_frames = len(frames_meta)
        T_augmented_world = self._sample_rigid()
        window_data = []
        for t in range(self.temporal_window):
            current_frame_idx = min(max(start_idx + t, 0), num_frames - 1)
            frame_meta = {**frames_meta[current_frame_idx], "sequence": sequence}
            if frames is None:
                graph = self._load_frame(frame_meta, T_augmented_world=T_augmented_world)
            else:
                key = (sequence, current_frame_idx)
                if key not in frames:
                    frames[key] = self._load_frame(frame_meta)
                graph = frames[key]
            window_data.append(graph)
        return window_data

    def _validate_graph(self, graph):
        if graph.edge_index.numel() > 0:
            assert int(graph.edge_index.min()) >= 0 and int(graph.edge_index.max()) < graph.x.size(0), (
                "Edge index out of bounds in SuperGraph"
            )
        assert graph.rank.size(0) == graph.x.size(0), "Rank tensor length mismatch"
        assert graph.t.size(0) == graph.x.size(0), "Time tensor length mismatch"
        if graph.t.numel() > 0:
            assert int(graph.t.min()) >= 0 and int(graph.t.max()) <= self.temporal_window - 1, "Time bounds incorrect"
        for name in ("screen_txt", "audio", "cam_params", "images"):
            assert getattr(graph, name).shape[0] == self.temporal_window
