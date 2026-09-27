from __future__ import annotations

import numpy as np
import torch

# Feature dims / column ids
RANK0_FEAT_DIM = 6  # [norm_x, norm_y, norm_z, joint_type, entity_id, node_type]
RANK1_FEAT_DIM = 5  # [rel_x, rel_y, rel_z, dist, edge_type]
RANK2_FEAT_DIM = 5  # [field0, field1, field2, field3, hyper_type]

R0_COL_JOINT_TYPE = 3
R0_COL_ENTITY_ID = 4
R0_COL_NODE_TYPE = 5

R1_COL_EDGE_TYPE = 4
R2_COL_HYPER_TYPE = 4
R2_COL_PERSON_ENTITY_ID = 3  # hypercell entity id (col 3 of x_2): person AND entity cells; -1 for triad/audio

# Hyper types
HYPER_TYPE_PERSON = 0
HYPER_TYPE_TRIAD = 1
HYPER_TYPE_AUDIO = 2

# Edge types
EDGE_TYPE_SKELETON = 0
EDGE_TYPE_SEMANTIC = 1
EDGE_TYPE_SPATIAL = 2
EDGE_TYPE_EVIDENCE_VISUAL = 3
EDGE_TYPE_EVIDENCE_TEXT = 4
EDGE_TYPE_EVIDENCE_JSON = 5
EDGE_TYPE_EVIDENCE_AUDIO = 6
EDGE_TYPE_AUDIO_BROADCAST = 7
EDGE_TYPE_EVIDENCE_SCREEN_IMAGE = 8

NUM_EDGE_TYPES = 9

# Node / evidence types
NODE_TYPE_JOINT = 0
NODE_TYPE_OBJECT = 1
NODE_TYPE_EVIDENCE = 2

EVIDENCE_SUBTYPE_VISUAL = 0
EVIDENCE_SUBTYPE_TEXT = 1
EVIDENCE_SUBTYPE_JSON = 2
EVIDENCE_SUBTYPE_AUDIO = 3
EVIDENCE_SUBTYPE_SCREEN_IMAGE = 4

NUM_JOINTS = 4

CAMERAS = (1, 2, 3, 4, 5)
CAMERA_IMAGE_SIZE = (2048, 1536)  # (width, height) of the recorded color images
INPUT_IMAGE_SIZE = (512, 384)  # (width, height) the visual backbone sees

# Object name to ID mapping
object_name_to_id = {
    "anaesthetist": 0,
    "anest": 0,  # Alias for anaesthetist
    "anesthesia_equipment": 1,
    "ae": 1,  # Alias for anesthesia_equipment
    "assistant_surgeon": 2,
    "c_arm": 3,
    "circulator": 4,
    "drape": 5,
    "drill": 6,
    "hammer": 7,
    "head_surgeon": 8,
    "instrument": 9,
    "instrument_table": 10,
    "mako_robot": 11,
    "monitor": 12,
    "mps": 13,
    "mps_station": 14,
    "nurse": 15,
    "operating_table": 16,
    "ot": 16,  # Alias for operating_table
    "patient": 17,
    "saw": 18,
    "secondary_table": 19,
    "student": 20,
    "tracker": 21,
    "unrelated_person": 22,
}

# Predicate name to ID mapping
predicate_name_to_id = {
    "assisting": 0,
    "calibrating": 1,
    "cementing": 2,
    "cleaning": 3,
    "closeto": 4,
    "cutting": 5,
    "drilling": 6,
    "hammering": 7,
    "holding": 8,
    "lyingon": 9,
    "manipulating": 10,
    "preparing": 11,
    "sawing": 12,
    "scanning": 13,
    "suturing": 14,
    "touching": 15,
    "none": 16,  # Added by the dataset code
}

NUM_PREDICATES = len(predicate_name_to_id)  # 17 (including "none")
PREDICATE_NONE_ID = predicate_name_to_id["none"]


def relation_labels_to_tensor(raw_relations: list) -> torch.Tensor:
    rows = []
    for triplet in raw_relations:
        sub_name, pred_name, obj_name = triplet
        sub_name = sub_name.lower().strip().replace(" ", "_")
        obj_name = obj_name.lower().strip().replace(" ", "_")
        pred_name = pred_name.lower().strip().replace(" ", "_")

        if (
            sub_name not in object_name_to_id
            or obj_name not in object_name_to_id
            or pred_name not in predicate_name_to_id
        ):
            continue

        rows.append(
            [
                object_name_to_id[sub_name],  # 0: Subject
                predicate_name_to_id[pred_name],  # 1: Predicate
                object_name_to_id[obj_name],  # 2: Object
            ]
        )

    if rows:
        return torch.tensor(rows, dtype=torch.long)
    return torch.zeros((0, 3), dtype=torch.long)


def next_action_to_label(raw_value):
    """Convert raw next_action JSON (None or ["name", seconds]) to label 0-11."""
    next_actions = [
        "bring in",
        "prepare",
        "clean",
        "cut",
        "drill",
        "saw",
        "hammer",
        "cement",
        "suture",
        "scan",
        "bring out",
        "none",
    ]

    if raw_value is None or len(raw_value) == 0:
        return 11  # 'none' action

    action_name = raw_value[0].lower().strip()

    if action_name in next_actions:
        return next_actions.index(action_name)
    else:
        raise ValueError(f"Unknown action: {action_name}")


def robot_phase_to_label(raw_value):
    """Convert raw robot_phase string to label 0-8."""
    robot_phases = [
        "turn on",
        "initial calibration by mps",
        "dressing the robot, to make it sterile",
        "install the saw by nurse",
        "install base array by nurse",
        "install calibration array",
        "calibrate the robot by nurse",
        "remove calibration array",
        "install actual saw tip",
    ]

    phase_name = raw_value.lower().strip()
    if phase_name in robot_phases:
        return robot_phases.index(phase_name)
    else:
        raise ValueError(f"Unknown robot phase: {phase_name}")


NUM_NEXT_ACTION_CLASSES = 12
NUM_ROBOT_PHASE_CLASSES = 9

# OR world-space normalization box (mm), shared by the scene-graph coordinate
# normalization and the model's de-normalization.
OR_CENTER = np.array([-500, 0, 1000], dtype=np.float32)
OR_HALF_SIZE = np.array([2500, 3000, 1000], dtype=np.float32)


# --- Human joint subset -----------------------------------------------------
# (name, index into the raw 15-keypoint panoptic skeleton). List order defines
# the "subset index" (0..N-1) used by every derived constant below.
JOINTS: list[tuple[str, int]] = [
    ("neck", 0),
    ("mid_hip", 2),
    ("left_wrist", 5),
    ("right_wrist", 11),
]

_SUBSET_IDX = {name: i for i, (name, _) in enumerate(JOINTS)}
_RAW_IDX = {name: raw for name, raw in JOINTS}

# Raw panoptic indices kept, in subset order — used to slice the (15, 4) pose array.
UPPER_BODY_INDICES: list[int] = [raw for _, raw in JOINTS]

# Subset indices (0..N-1) of named joints.
HEAD_GRAPH_INDEX: int = _SUBSET_IDX["neck"]
PELVIS_IDX: int = _SUBSET_IDX["mid_hip"]
WRIST_GRAPH_INDICES: list[int] = [_SUBSET_IDX["left_wrist"], _SUBSET_IDX["right_wrist"]]
# subset index -> raw panoptic index, for the two wrists.
WRIST_RAW_MAP: dict[int, int] = {_SUBSET_IDX[n]: _RAW_IDX[n] for n in ("left_wrist", "right_wrist")}

# Intra-person skeleton edges in subset indices (star from the pelvis/root).
SKELETON_EDGES: list[tuple[int, int]] = [
    (PELVIS_IDX, _SUBSET_IDX[n]) for n in ("neck", "left_wrist", "right_wrist")
]

# Minimum keypoint confidence to instantiate a joint node.
CONF_THRESHOLD: float = 0.01

# Distance scales (mm): spatial edges connect entities closer than SPATIAL_THRESHOLD_MM,
# and edge features store distance / scale clipped to [0, 1].
SPATIAL_THRESHOLD_MM: float = 500.0
SEMANTIC_THRESHOLD_MM: float = 4 * SPATIAL_THRESHOLD_MM

assert len(UPPER_BODY_INDICES) == NUM_JOINTS, (
    f"UPPER_BODY_INDICES has {len(UPPER_BODY_INDICES)} entries but NUM_JOINTS={NUM_JOINTS}"
)

# --- Entity IDs ---
ENTITY_HEAD_SURGEON = object_name_to_id["head_surgeon"]
ENTITY_ASSISTANT_SURGEON = object_name_to_id["assistant_surgeon"]
ENTITY_NURSE = object_name_to_id["nurse"]
ENTITY_CIRCULATOR = object_name_to_id["circulator"]
ENTITY_STUDENT = object_name_to_id["student"]
ENTITY_ANESTHETIST = object_name_to_id["anaesthetist"]
ENTITY_MPS = object_name_to_id["mps"]
ENTITY_UNRELATED_PERSON = object_name_to_id["unrelated_person"]
ENTITY_PATIENT = object_name_to_id["patient"]
ENTITY_DRILL = object_name_to_id["drill"]
ENTITY_HAMMER = object_name_to_id["hammer"]
ENTITY_INSTRUMENT = object_name_to_id["instrument"]
ENTITY_SAW = object_name_to_id["saw"]
ENTITY_INSTRUMENT_TABLE = object_name_to_id["instrument_table"]
ENTITY_MPS_STATION = object_name_to_id["mps_station"]
ENTITY_MAKO_ROBOT = object_name_to_id["mako_robot"]

HUMAN_ENTITIES = {
    ENTITY_HEAD_SURGEON,
    ENTITY_ASSISTANT_SURGEON,
    ENTITY_NURSE,
    ENTITY_MPS,
    ENTITY_CIRCULATOR,
    ENTITY_STUDENT,
    ENTITY_ANESTHETIST,
    ENTITY_UNRELATED_PERSON,
}

SEMANTIC_EDGES = [
    (ENTITY_HEAD_SURGEON, ENTITY_DRILL),
    (ENTITY_HEAD_SURGEON, ENTITY_HAMMER),
    (ENTITY_HEAD_SURGEON, ENTITY_INSTRUMENT),
    (ENTITY_HEAD_SURGEON, ENTITY_SAW),
    (ENTITY_HEAD_SURGEON, ENTITY_MAKO_ROBOT),
    (ENTITY_ASSISTANT_SURGEON, ENTITY_MAKO_ROBOT),
    (ENTITY_MPS, ENTITY_MPS_STATION),
    (ENTITY_MPS, ENTITY_MAKO_ROBOT),
    (ENTITY_NURSE, ENTITY_MAKO_ROBOT),
    (ENTITY_NURSE, ENTITY_INSTRUMENT_TABLE),
]

FUNCTIONAL_TRIADS = [
    (
        [ENTITY_HEAD_SURGEON],
        [ENTITY_DRILL, ENTITY_HAMMER, ENTITY_INSTRUMENT, ENTITY_MAKO_ROBOT],
        [ENTITY_PATIENT, ENTITY_ASSISTANT_SURGEON],
    ),
    (
        [ENTITY_MPS],
        [ENTITY_MAKO_ROBOT],
        [ENTITY_NURSE],
    ),
]
