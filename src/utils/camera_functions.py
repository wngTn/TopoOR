import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation as R

CAM_PARAM_MAP: dict[str, tuple[slice, tuple[int, ...]]] = {
    "R_camera_world": (slice(0, 9), (3, 3)),
    "R_world_camera": (slice(9, 18), (3, 3)),
    "t_camera_world": (slice(18, 21), (3,)),
    "t_world_camera": (slice(21, 24), (3,)),
    "K": (slice(24, 33), (3, 3)),
    "K_inv": (slice(33, 42), (3, 3)),
    "k": (slice(42, 48), (6,)),
    "p": (slice(48, 50), (2,)),
    "cam_idx": (slice(50, 51), (1,)),
}
"\nTotal elements: 9+9+3+3+9+9+6+2+1 = 51\n"


def get_camera_params(cam_vector: torch.Tensor, keys: list[str]) -> list[torch.Tensor]:
    """Extract ``keys`` from a flattened (*, 51) camera tensor, reshaped to (*, 3, 3)/(*, 3)."""
    if cam_vector.shape[-1] != 51:
        raise ValueError(f"Input vector's last dimension must be 51, but got shape {cam_vector.shape}")
    output_params = []
    for key in keys:
        if key not in CAM_PARAM_MAP:
            raise KeyError(f"Unknown camera parameter key: '{key}'. Valid keys are: {list(CAM_PARAM_MAP.keys())}")
        slice_obj, target_shape = CAM_PARAM_MAP[key]
        sliced_param = cam_vector[..., slice_obj]
        if len(target_shape) == 2:
            h, w = target_shape
            reshaped_param = rearrange(sliced_param, "... (h w) -> ... h w", h=h, w=w)
        else:
            reshaped_param = sliced_param
        output_params.append(reshaped_param)
    return output_params


def world_3d_to_img_2d(X: torch.Tensor, cam_params_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D world points X into pixel coordinates using the Rational Distortion Model.
    Includes dimension-agnostic safety clamps to prevent NaN/Inf during training.
    """
    if cam_params_vec.ndim < X.ndim - 1:
        need = X.ndim - 1 - cam_params_vec.ndim
        cam_params_vec = cam_params_vec.view((1,) * need + cam_params_vec.shape)
    R_camera_world, t_camera_world, K, k, p = get_camera_params(
        cam_params_vec, ["R_camera_world", "t_camera_world", "K", "k", "p"]
    )
    X_cam = torch.einsum("...vcd,...njd->...njvc", R_camera_world, X)
    X_cam = X_cam + t_camera_world[..., None, None, :, :]
    z = X_cam[..., 2:3]
    valid_z = z > 1e-06
    z = z.clamp_min(1e-06)
    normalized = X_cam[..., :2] / z
    normalized = torch.where(valid_z, normalized, torch.zeros_like(normalized))
    x_n = normalized[..., 0]
    y_n = normalized[..., 1]
    r2 = x_n * x_n + y_n * y_n
    r2 = r2.clamp(max=25.0)
    r4 = r2 * r2
    r6 = r4 * r2
    k1 = k[..., 0][..., None, None, :]
    k2 = k[..., 1][..., None, None, :]
    k3 = k[..., 2][..., None, None, :]
    k4 = k[..., 3][..., None, None, :]
    k5 = k[..., 4][..., None, None, :]
    k6 = k[..., 5][..., None, None, :]
    p1 = p[..., 0][..., None, None, :]
    p2 = p[..., 1][..., None, None, :]
    numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    denominator = torch.where(torch.abs(denominator) < 1e-06, torch.sign(denominator + 1e-09) * 1e-06, denominator)
    radial = numerator / denominator
    radial = torch.clamp(radial, min=-20.0, max=20.0)
    tan_x = 2.0 * p1 * x_n * y_n + p2 * (r2 + 2.0 * x_n * x_n)
    tan_y = p1 * (r2 + 2.0 * y_n * y_n) + 2.0 * p2 * x_n * y_n
    xd = x_n * radial + tan_x
    yd = y_n * radial + tan_y
    distorted_norm = torch.stack([xd, yd], dim=-1)
    f = torch.stack([K[..., 0, 0], K[..., 1, 1]], dim=-1)[..., None, None, :, :]
    c = K[..., :2, 2][..., None, None, :, :]
    pix = distorted_norm * f + c
    pix = rearrange(pix, "... n j v c -> ... v n j c")
    valid_mask = rearrange(valid_z, "... n j v c -> ... v n j c")
    return (pix, valid_mask)


def image_affine(original_size, input_size) -> np.ndarray:
    """2x3 affine taking ``original_size`` image pixels to ``input_size`` pixels.

    The map is the least-squares fit through three corresponding points of the two
    images: the centre, the point half the image width above it, and that offset
    turned by 90 degrees. The original width is rounded to float32 before the fit.
    """

    def triangle(size, width):
        center = torch.tensor([[size[0] / 2, size[1] / 2]], dtype=torch.float64)
        top = center + torch.stack([torch.zeros_like(width), -0.5 * width], dim=-1)
        offset = center - top
        return torch.stack([center, top, top + torch.stack([-offset[..., 1], offset[..., 0]], dim=-1)], dim=-2)

    original_width = torch.tensor([np.float32(original_size[0] / 200.0)], dtype=torch.float64) * 200.0
    source = triangle(original_size, original_width)
    target = triangle(input_size, torch.tensor([float(input_size[0])], dtype=torch.float64))
    homogeneous = torch.cat([source, torch.ones_like(source[..., :1])], dim=-1)
    return (torch.linalg.pinv(homogeneous) @ target).transpose(-1, -2).numpy()[0]


def get_cam_params(sequences: list[str], data_root: Path, cameras: list[int]) -> dict[str, np.ndarray]:
    cam_params = {}
    T_color_camera = np.eye(4, dtype=np.float32)
    T_color_camera[:3, :3] = R.from_euler("x", 180, degrees=True).as_matrix()
    T_world_raw = np.eye(4, dtype=np.float32)
    T_world_raw[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix()
    SCALE_M_TO_MM = 1000.0
    for sequence in sequences:
        sequence_path = data_root / sequence
        cam_params_seq = []
        for cam_idx in sorted(cameras):
            cal_path = sequence_path / f"camera{cam_idx:02d}.json"
            with open(cal_path, "r") as f:
                data = json.load(f)["value0"]
            t_dict = data["camera_pose"]["translation"]
            t_raw = np.array([t_dict["m00"], t_dict["m10"], t_dict["m20"]], dtype=np.float32) * SCALE_M_TO_MM
            q_raw = [data["camera_pose"]["rotation"][k] for k in ["x", "y", "z", "w"]]
            r_raw = R.from_quat(q_raw).as_matrix().astype(np.float32)
            T_raw_depth = np.eye(4, dtype=np.float32)
            T_raw_depth[:3, :3] = r_raw
            T_raw_depth[:3, 3] = t_raw
            c2d_data = data.get("color2depth_transform", None)
            if c2d_data:
                t_c2d_dict = c2d_data["translation"]
                t_c2d = (
                    np.array([t_c2d_dict["m00"], t_c2d_dict["m10"], t_c2d_dict["m20"]], dtype=np.float32)
                    * SCALE_M_TO_MM
                )
                q_c2d = [
                    c2d_data["rotation"]["x"],
                    c2d_data["rotation"]["y"],
                    c2d_data["rotation"]["z"],
                    c2d_data["rotation"]["w"],
                ]
                r_c2d = R.from_quat(q_c2d).as_matrix().astype(np.float32)
                T_depth_color = np.eye(4, dtype=np.float32)
                T_depth_color[:3, :3] = r_c2d
                T_depth_color[:3, 3] = t_c2d
                T_raw_color = T_raw_depth @ T_depth_color
            else:
                T_raw_color = T_raw_depth
            T_world_camera = T_world_raw @ T_raw_color @ T_color_camera
            T_camera_world = np.linalg.inv(T_world_camera)
            R_camera_world = T_camera_world[:3, :3]
            t_camera_world = T_camera_world[:3, 3]
            R_world_camera = T_world_camera[:3, :3]
            t_world_camera = T_world_camera[:3, 3]
            i_dict = data["color_parameters"]["intrinsics_matrix"]
            K = np.array(
                [
                    [i_dict["m00"], i_dict["m10"], i_dict["m20"]],
                    [i_dict["m01"], i_dict["m11"], i_dict["m21"]],
                    [i_dict["m02"], i_dict["m12"], i_dict["m22"]],
                ],
                dtype=np.float32,
            )
            K[2, 1], K[2, 2] = (0.0, 1.0)
            K_inv = np.linalg.inv(K)
            rad_dict = data["color_parameters"]["radial_distortion"]
            tan_dict = data["color_parameters"]["tangential_distortion"]
            k = np.array(
                [
                    rad_dict.get("k1", 0),
                    rad_dict.get("k2", 0),
                    rad_dict.get("k3", 0),
                    rad_dict.get("k4", 0),
                    rad_dict.get("k5", 0),
                    rad_dict.get("k6", 0),
                ],
                dtype=np.float32,
            )
            p = np.array([tan_dict.get("p1", 0), tan_dict.get("p2", 0)], dtype=np.float32)
            cam_params_v = np.hstack(
                [
                    R_camera_world.flatten(),
                    R_world_camera.flatten(),
                    t_camera_world.flatten(),
                    t_world_camera.flatten(),
                    K.flatten(),
                    K_inv.flatten(),
                    k.flatten(),
                    p.flatten(),
                    np.array([cam_idx], dtype=np.float32),
                ]
            )
            cam_params_seq.append(cam_params_v)
        cam_params[sequence] = np.stack(cam_params_seq)
    return cam_params
