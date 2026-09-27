from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.data.datasets.constants import (
    CAMERAS,
    next_action_to_label,
    relation_labels_to_tensor,
    robot_phase_to_label,
)
from src.data.datasets.modality_io import PACKED_DIR


def timestamps_path(data_root: Path, sequence: str) -> Path:
    name = (
        "timestamp_to_pcd_and_frames_list_azure.json"
        if sequence == "010_PKA"
        else "timestamp_to_pcd_and_frames_list.json"
    )
    return data_root / sequence / name


# Bump when the cached index format changes.
INDEX_CACHE_VERSION = 3


def index_cache_directory():
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "topoor" / "mm_or_index_cache"


class IndexCacheMixin:
    """Disk cache for the per-frame MM-OR index."""

    def _index_cache_path(self) -> Path:
        """Deterministic cache file path for the current (split, config) index."""
        key = {
            "version": INDEX_CACHE_VERSION,
            "data_root": str(self.data_root.resolve()),
            "packed_dir": str(PACKED_DIR.resolve()),
            "sequences": sorted(self.sequences),
            "temporal_window": self.temporal_window,
            "stride": self.stride,
            "split": self.split,
            "max_frames_per_seq": self.max_frames_per_seq,
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        return index_cache_directory() / f"index_{self.split}_{digest}.pkl"

    def _input_fingerprint(self) -> dict:
        """Detect changed inputs from file sizes and modification times."""

        def sig(p: Path):
            try:
                st = p.stat()
                return [st.st_mtime_ns, st.st_size]
            except FileNotFoundError:
                return None

        fp: dict = {}
        for seq in sorted(self.sequences):
            fp[seq] = {
                "timestamps": sig(timestamps_path(self.data_root, seq)),
                "next_action": sig(self.data_root / "take_timestamp_to_next_action" / f"{seq}.json"),
                "robot_phase": sig(self.data_root / "take_timestamp_to_robot_phase" / f"{seq}.json"),
                "sterility": sig(self.data_root / "take_timestamp_to_sterility_breach" / f"{seq}.json"),
                "relation_dir": sig(self.data_root / seq / "relation_labels"),
            }
        # Repacking modalities must invalidate the index too.
        fp["__packed__"] = {p.name: sig(p) for p in sorted(PACKED_DIR.glob("*.pkl"))}
        return fp

    def _load_index_cache(self, cache_path: Path, fingerprint: dict):
        """Return (sequence_frames, samples) from cache iff present and fresh."""
        if not cache_path.is_file():
            return None
        try:
            with cache_path.open("rb") as fh:
                blob = pickle.load(fh)
        except Exception as e:  # corrupt / version-mismatched cache -> rebuild
            self.log.warning(f"[MMORDataset] Ignoring unreadable index cache {cache_path}: {e}")
            return None
        if blob.get("fingerprint") != fingerprint:
            return None
        return blob["sequence_frames"], blob["samples"]

    def _save_index_cache(self, cache_path: Path, fingerprint: dict) -> None:
        temporary = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=cache_path.parent, prefix=f".{cache_path.name}.", delete=False) as fh:
                temporary = Path(fh.name)
                pickle.dump(
                    {
                        "fingerprint": fingerprint,
                        "sequence_frames": self.sequence_frames,
                        "samples": self.samples,
                    },
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(temporary, cache_path)
            self.log.info(f"[MMORDataset] Saved index cache -> {cache_path}")
        except Exception as e:
            self.log.warning(f"[MMORDataset] Could not write index cache {cache_path}: {e}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class IndexBuildMixin:
    """Build frame metadata and temporal windows from disk."""

    def _build_index(self, use_cache):
        """Load the per-frame index from disk cache when fresh, else build + cache it."""
        cache_path = fingerprint = None
        if use_cache:
            cache_path = self._index_cache_path()
            fingerprint = self._input_fingerprint()
            cached = self._load_index_cache(cache_path, fingerprint)
            if cached is not None:
                self.sequence_frames, self.samples = cached
                self.log.info(
                    f"[MMORDataset] split={self.split}: loaded index from cache ({len(self.samples)} samples, {len(self.sequence_frames)} seqs) <- {cache_path}"
                )
                return
        self._build_index_uncached()
        if use_cache:
            self._save_index_cache(cache_path, fingerprint)

    def _build_index_uncached(self):
        self.sequence_frames = {}
        for sequence in self.sequences:
            frames = self._sequence_index(sequence)
            if frames:
                self.sequence_frames[sequence] = frames
        self.samples = []
        for sequence, frames in self.sequence_frames.items():
            num_frames = len(frames)
            if self.stride > 1:
                for start_idx in range(0, num_frames - self.temporal_window + 1, self.stride):
                    self.samples.append((sequence, start_idx))
                last_start = num_frames - self.temporal_window
                if last_start >= 0 and (not self.samples or self.samples[-1] != (sequence, last_start)):
                    self.samples.append((sequence, last_start))
            else:
                start_range_begin = -(self.temporal_window - 1)
                start_range_end = num_frames - self.temporal_window + 1
                for start_idx in range(start_range_begin, start_range_end):
                    self.samples.append((sequence, start_idx))
        self.log.info(
            f"[MMORDataset] split={self.split}, sequences={len(self.sequence_frames)}, samples={len(self.samples)}, window={self.temporal_window}, stride={self.stride}"
        )

    def _sequence_index(self, sequence):
        path_to_seq = self.data_root / sequence
        colordir = path_to_seq / "colorimage"
        try:
            colorimage_existing = set(os.listdir(colordir))
        except FileNotFoundError:
            colorimage_existing = set()
            self.log.warning(f"[MMORDataset] colorimage dir missing for sequence={sequence}: {colordir}")
        bboxes = self._bbox_packed.get(sequence, {})
        poses = self._pose_packed.get(sequence, {})
        if not bboxes or not poses:
            return []
        time_stamps = json.load(timestamps_path(self.data_root, sequence).open("r"))
        if self.max_frames_per_seq is not None:
            time_stamps = time_stamps[: self.max_frames_per_seq]
        next_action_data = json.load((self.data_root / "take_timestamp_to_next_action" / f"{sequence}.json").open("r"))
        robot_phase_data = json.load((self.data_root / "take_timestamp_to_robot_phase" / f"{sequence}.json").open("r"))
        sterility_breach_data = json.load(
            (self.data_root / "take_timestamp_to_sterility_breach" / f"{sequence}.json").open("r")
        )
        frames = []
        for idx, (_, mapping, *_) in enumerate(time_stamps):
            rgb_frame_idx = int(mapping["azure"])
            simstation_frame_idx = int(mapping["simstation"]) if mapping["simstation"] is not None else -1
            colorimage_paths = []
            for cam_idx in CAMERAS:
                color_path = colordir / f"camera{cam_idx:02d}_colorimage-{rgb_frame_idx:06d}.jpg"
                if color_path.name not in colorimage_existing:
                    self.log.warning(
                        f"[MMORDataset] Missing color image for sequence={sequence}, cam{cam_idx}, frame_idx={rgb_frame_idx}"
                    )
                colorimage_paths.append(color_path)
            bbox_key = rgb_frame_idx
            pose_key = rgb_frame_idx
            audio_key = idx
            txt_key = simstation_frame_idx
            json_key = simstation_frame_idx
            screen_img_file = path_to_seq / "simstation" / f"camera01_{simstation_frame_idx:06d}.jpg"
            if bbox_key not in bboxes:
                self.log.warning(f"[MMORDataset] Missing bbox for sequence={sequence}, frame_idx={rgb_frame_idx}")
                continue
            if pose_key not in poses:
                self.log.warning(f"[MMORDataset] Missing pose for sequence={sequence}, frame_idx={rgb_frame_idx}")
                continue
            if (
                f"{idx:06d}" not in next_action_data
                or f"{idx:06d}" not in robot_phase_data
                or f"{idx:06d}" not in sterility_breach_data
            ):
                self.log.warning(f"[MMORDataset] Missing labels for sequence={sequence}, frame_idx={idx}")
                continue
            relation_labels_path = self.data_root / sequence / "relation_labels" / f"{idx:06d}.json"
            raw_relations = (
                json.load(relation_labels_path.open("r"))["rel_annotations"] if relation_labels_path.is_file() else []
            )
            relation_labels_tensor = relation_labels_to_tensor(raw_relations)
            frames.append(
                {
                    "idx": idx,
                    "frame_idx": rgb_frame_idx,
                    "colorimage_paths": colorimage_paths,
                    "bbox_key": bbox_key,
                    "pose_key": pose_key,
                    "txt_key": txt_key,
                    "json_key": json_key,
                    "audio_key": audio_key,
                    "screen_img_path": screen_img_file,
                    "next_action": next_action_to_label(next_action_data[f"{idx:06d}"]),
                    "robot_phase": robot_phase_to_label(robot_phase_data[f"{idx:06d}"]),
                    "relation_labels": relation_labels_tensor,
                }
            )
        return frames
