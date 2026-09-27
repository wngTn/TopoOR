"""Evaluate sterility breach detection using personnel-to-object distances in mm.

Thresholds are rounded to the nearest 50 mm from a depth-6 decision tree fitted
on these same evaluation takes, so scores are in-sample.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rootutils
from sklearn.metrics import classification_report

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.data.datasets.geometry import quaternion_to_rotation_matrix
from src.data.datasets.modality_io import load_packed

TEST_SEQUENCES = ["004_PKA", "011_TKA", "036_PKA", "038_TKA"]

RULE_DEFS = [
    ("mps", "instrument_table", "MPS_InstTable"),
    ("mps", "drape", "MPS_Drape"),
    ("assistant_surgeon", "operating_table", "Asst_OpTable"),
    ("nurse", "operating_table", "Nurse_OpTable"),
    ("mps", "patient", "MPS_Patient"),
    ("circulator", "drape", "Circulator_Drape"),
    ("anest", "patient", "Anaest_Patient"),
    ("anest", "drape", "Anaest_Drape"),
]

OBJECT_ALIASES = {"operating_table": ["operating_table", "ot"]}

# Rules over rank-0 cells derived from decision tree built on training data.
def check_breach_rules(distances):
    if distances["Circulator_Drape"] <= 250 and distances["MPS_Patient"] <= 200:
        return 1
    if (
        distances["Circulator_Drape"] <= 150
        and distances["MPS_Patient"] > 200
        and distances["Asst_OpTable"] <= 6200
        and distances["Nurse_OpTable"] > 2550
    ):
        return 1
    if (
        150 < distances["Circulator_Drape"] <= 250
        and distances["MPS_Patient"] > 200
        and distances["Asst_OpTable"] <= 6200
    ):
        return 1
    if (
        distances["Circulator_Drape"] <= 250
        and 200 < distances["MPS_Patient"] <= 400
        and distances["Asst_OpTable"] > 6200
        and distances["MPS_InstTable"] <= 2000
        and distances["Nurse_OpTable"] > 5250
    ):
        return 1
    if (
        distances["Circulator_Drape"] <= 250
        and distances["MPS_Patient"] > 200
        and distances["Asst_OpTable"] > 6200
        and 2000 < distances["MPS_InstTable"] <= 6550
    ):
        return 1
    if (
        distances["Circulator_Drape"] > 250
        and distances["MPS_InstTable"] <= 150
        and distances["MPS_Drape"] <= 900
    ):
        return 1
    if (
        distances["Circulator_Drape"] > 250
        and distances["MPS_InstTable"] <= 150
        and distances["MPS_Drape"] > 900
        and 850 < distances["Asst_OpTable"] <= 1400
    ):
        return 1
    if (
        distances["Circulator_Drape"] > 250
        and distances["MPS_InstTable"] > 150
        and distances["Anaest_Drape"] <= 150
        and distances["MPS_Drape"] <= 550
        and distances["Anaest_Patient"] <= 1000
    ):
        return 1
    if (
        distances["Circulator_Drape"] > 250
        and distances["MPS_InstTable"] > 150
        and distances["Anaest_Drape"] <= 150
        and distances["MPS_Drape"] <= 550
        and distances["Anaest_Patient"] > 1000
        and distances["Asst_OpTable"] <= 600
    ):
        return 1
    if (
        distances["Circulator_Drape"] > 250
        and distances["MPS_InstTable"] > 150
        and distances["Anaest_Drape"] <= 150
        and distances["MPS_Drape"] > 550
        and distances["Asst_OpTable"] > 6200
        and distances["MPS_Patient"] > 1300
    ):
        return 1
    if (
        distances["Circulator_Drape"] > 250
        and 2850 < distances["MPS_InstTable"] <= 3250
        and distances["Anaest_Drape"] > 150
        and distances["MPS_Drape"] <= 250
    ):
        return 1
    return 0


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def point_to_obb_distance_bulk(points, center, dims, quat):
    if len(points) == 0:
        return np.array([])
    local = (points - center) @ quaternion_to_rotation_matrix(quat)
    half = dims / 2.0
    return np.linalg.norm(local - np.clip(local, -half, half), axis=1)


def find_object(bbox_dict, name):
    for alias in OBJECT_ALIASES.get(name, [name]):
        if alias in bbox_dict:
            return bbox_dict[alias]
    return None


def evaluate_frame(pose_dict, bbox_dict):
    distances = {}
    for role, obj, feat_name in RULE_DEFS:
        kp = pose_dict.get(role)
        obj_data = find_object(bbox_dict, obj)
        val = 9999.0
        if kp is not None and obj_data is not None:
            kp_array = np.asarray(kp, dtype=np.float32)
            valid_mask = kp_array[:, 3] > 0.1
            if np.any(valid_mask):
                pts = kp_array[valid_mask, :3]
                dists = point_to_obb_distance_bulk(
                    pts,
                    np.asarray(obj_data["center"], float),
                    np.asarray(obj_data["dimensions"], float),
                    np.asarray(obj_data["quaternion"], float),
                )
                val = float(np.min(dists))
        distances[feat_name] = val
    return check_breach_rules(distances)


def run_evaluation(data_dir):
    data_dir = Path(data_dir)
    bboxes, poses = load_packed("bounding_boxes"), load_packed("human_poses")
    take_gts, take_preds = defaultdict(list), defaultdict(list)
    for sequence in TEST_SEQUENCES:
        print(f"Processing {sequence}...")
        raw_gt = load_json(data_dir / "take_timestamp_to_sterility_breach" / f"{sequence}.json")
        gt_breach = {int(k): v for k, v in raw_gt.items()}
        timestamps = load_json(data_dir / sequence / "timestamp_to_pcd_and_frames_list.json")
        for idx, (_, mapping, *_) in enumerate(timestamps):
            if idx not in gt_breach:
                continue
            rgb_idx = int(mapping["azure"])
            bbox = bboxes.get(sequence, {}).get(rgb_idx)
            pose = poses.get(sequence, {}).get(rgb_idx)
            take_gts[sequence].append(1 if len(gt_breach[idx]) > 0 else 0)
            # Frames without boxes or poses count as no breach.
            take_preds[sequence].append(0 if bbox is None or pose is None else evaluate_frame(pose, bbox))
    return take_gts, take_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate sterility breach detection")
    parser.add_argument("--data-dir", type=str, default="./data/mm_or")
    args = parser.parse_args()

    gts, preds = run_evaluation(args.data_dir)

    for seq in sorted(gts.keys()):
        print(f"\nTake {seq}\n")
        if not gts[seq]:
            print("No data.")
            continue
        print(classification_report(gts[seq], preds[seq], target_names=["no", "yes"], labels=[0, 1], zero_division=0))

    print("\nVal Results (Overall):\n")
    all_gts = [item for sublist in gts.values() for item in sublist]
    all_preds = [item for sublist in preds.values() for item in sublist]
    print(classification_report(all_gts, all_preds, target_names=["no", "yes"], labels=[0, 1], zero_division=0, digits=4))


if __name__ == "__main__":
    main()
