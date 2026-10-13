"""Reward formulas for HUMOTO body, hand and object tracking.

The reward manager multiplies each value by its preset weight and control dt.
Tracking kernels use exp(-squared_error / sigma**2); sigma is in metres for
positions and radians for joint angles. Preset weights live in
wuji_wbt.configs.reward_wuji. Positions/quaternions below use the motion
loader's world coordinates and xyzw quaternion convention.
"""

from __future__ import annotations

from holosoma.managers.reward.terms.wbt import _get_motion_command_and_assert_type
from holosoma.utils.rotations import quat_rotate_inverse
from holosoma.utils.safe_torch_import import torch

PALMS = ["wj_left_palm_link", "wjr_right_palm_link"]
TIPS = {
    "wj_left_palm_link": [f"wj_left_finger{f}_link4" for f in range(1, 6)],
    "wjr_right_palm_link": [f"wjr_right_finger{f}_link4" for f in range(1, 6)],
}


def _cached(env, key, build):
    v = getattr(env, key, None)
    if v is None:
        v = build()
        setattr(env, key, v)
    return v


def _finger_dof_idx(env):
    """Indices of the 40 finger DOFs in the robot's dof order (robust to ordering)."""
    return _cached(
        env,
        "_wuji_finger_dof_idx",
        lambda: torch.tensor(
            [i for i, n in enumerate(env.dof_names) if "finger" in n], device=env.device, dtype=torch.long
        ),
    )


def _body_dof_mask(env):
    return _cached(
        env,
        "_wuji_body_dof_mask",
        lambda: torch.tensor(["finger" not in n for n in env.dof_names], device=env.device, dtype=torch.bool),
    )


def _hand_body_idx(env):
    """(palm_idx [2], tip_idx [2, 5]) into the simulator body list = the order the motion loader uses."""

    def build():
        names = list(env.simulator._body_list)
        palm = torch.tensor([names.index(p) for p in PALMS], device=env.device, dtype=torch.long)
        tip = torch.tensor([[names.index(t) for t in TIPS[p]] for p in PALMS], device=env.device, dtype=torch.long)
        return palm, tip

    return _cached(env, "_wuji_hand_body_idx", build)


def finger_joint_pos_error_exp(env, sigma: float) -> torch.Tensor:
    """Track all 40 finger angles: exp(-mean squared angular error / sigma²)."""
    mc = _get_motion_command_and_assert_type(env)
    idx = _finger_dof_idx(env)
    err = torch.square(mc.joint_pos[:, idx] - env.simulator.dof_pos[:, idx]).mean(-1)
    return torch.exp(-err / sigma**2)


def _tips_in_palm(pos_w, quat_w_xyzw, palm, tip):
    """pos_w [N, B, 3], quat [N, B, 4] xyzw -> the 10 fingertip positions in their palm frame [N, 10, 3]."""
    n, k = pos_w.shape[0], tip.shape[1]
    p_palm = pos_w[:, palm][:, :, None, :]  # [N, 2, 1, 3]
    q_palm = quat_w_xyzw[:, palm][:, :, None, :].expand(-1, -1, k, -1)  # [N, 2, 5, 4]
    rel = pos_w[:, tip] - p_palm  # [N, 2, 5, 3]
    local = quat_rotate_inverse(q_palm.reshape(-1, 4), rel.reshape(-1, 3), w_last=True)
    return local.reshape(n, -1, 3)


def fingertip_pos_palm_error_exp(env, sigma: float) -> torch.Tensor:
    """Track ten fingertip-link positions in their own palm frames; sigma is in metres."""
    mc = _get_motion_command_and_assert_type(env)
    palm, tip = _hand_body_idx(env)
    t = mc.time_steps
    ref = _tips_in_palm(mc.motion.body_pos_w[t], mc.motion.body_quat_w[t], palm, tip)
    cur = _tips_in_palm(env.simulator._rigid_body_pos, env.simulator._rigid_body_rot, palm, tip)
    err = torch.sum(torch.square(ref - cur), dim=-1).mean(-1)
    return torch.exp(-err / sigma**2)


def _tips_in_object(pos_w, obj_pos_w, obj_quat_w_xyzw, tip):
    """pos_w [N, B, 3] -> the 10 fingertip positions in the object frame [N, 10, 3]."""
    n, k = pos_w.shape[0], tip.numel()
    rel = pos_w[:, tip.reshape(-1)] - obj_pos_w[:, None, :]  # [N, 10, 3]
    q = obj_quat_w_xyzw[:, None, :].expand(-1, k, -1)
    return quat_rotate_inverse(q.reshape(-1, 4), rel.reshape(-1, 3), w_last=True).reshape(n, k, 3)


def fingertip_pos_object_error_exp(env, sigma: float) -> torch.Tensor:
    """Track ten fingertip-link positions relative to the object; sigma is in metres."""
    mc = _get_motion_command_and_assert_type(env)
    _, tip = _hand_body_idx(env)
    t = mc.time_steps
    ref = _tips_in_object(mc.motion.body_pos_w[t], mc.motion.object_pos_w[t], mc.motion.object_quat_w[t], tip)
    cur = _tips_in_object(env.simulator._rigid_body_pos, mc.simulator_object_pos_w, mc.simulator_object_quat_w, tip)
    err = torch.sum(torch.square(ref - cur), dim=-1).mean(-1)
    return torch.exp(-err / sigma**2)


def _ref_tip_contact(env):
    """Reference per-frame fingertip contact flags [T, 10] (bool), loaded once from <motion>_tipcontact.npy."""

    def build():
        from pathlib import Path

        import numpy as np

        from holosoma.utils.path import resolve_data_file_path

        mc = _get_motion_command_and_assert_type(env)
        mf = Path(resolve_data_file_path(mc.motion_cfg.motion_file))
        flags = np.load(mf.with_name(mf.stem + "_tipcontact.npy"))
        return torch.tensor(flags, device=env.device, dtype=torch.float32)

    return _cached(env, "_wuji_ref_tip_contact", build)


def fingertip_contact(env, force_thresh: float = 1.0, near_ref: float = 0.05) -> torch.Tensor:
    """Fraction of reference-contacting tips passing force and proximity thresholds.

    Force is the net link force, so proximity to the reference in object
    coordinates gates out distant self-contact. This is a contact proxy;
    it does not identify the other colliding body. Returns zero when the
    reference has no active contacts. Threshold units are newtons/metres."""
    mc = _get_motion_command_and_assert_type(env)
    _, tip = _hand_body_idx(env)
    t = mc.time_steps
    ref = _ref_tip_contact(env)[t.clamp(max=_ref_tip_contact(env).shape[0] - 1)]  # [N, 10]
    force = torch.norm(env.simulator.contact_forces[:, tip.reshape(-1)], dim=-1)  # [N, 10]
    p_ref = _tips_in_object(mc.motion.body_pos_w[t], mc.motion.object_pos_w[t], mc.motion.object_quat_w[t], tip)
    p_cur = _tips_in_object(env.simulator._rigid_body_pos, mc.simulator_object_pos_w, mc.simulator_object_quat_w, tip)
    on_object = torch.norm(p_cur - p_ref, dim=-1) < near_ref  # [N, 10]
    sim = ((force > force_thresh) & on_object).float()
    return (ref * sim).sum(-1) / ref.sum(-1).clamp(min=1.0)


# every collidable hand link (palm + finger link2/3/4), 16 per hand; order shared with tools/make_link_contact.py
HAND_LINKS = [
    f"wj_left_{n}"
    for n in ["palm_link"] + [f"finger{finger}_link{link}" for finger in range(1, 6) for link in (2, 3, 4)]
] + [
    f"wjr_right_{n}"
    for n in ["palm_link"] + [f"finger{finger}_link{link}" for finger in range(1, 6) for link in (2, 3, 4)]
]


def _hand_link_idx(env):
    def build():
        names = list(env.simulator._body_list)
        return torch.tensor([names.index(n) for n in HAND_LINKS], device=env.device, dtype=torch.long)

    return _cached(env, "_wuji_hand_link_idx", build)


def _ref_link_contact(env):
    """Reference per-frame hand-link contact flags [T, 32] (tools/make_link_contact.py -> <motion>_linkcontact.npz)."""

    def build():
        from pathlib import Path

        import numpy as np

        from holosoma.utils.path import resolve_data_file_path

        mc = _get_motion_command_and_assert_type(env)
        mf = Path(resolve_data_file_path(mc.motion_cfg.motion_file))
        flags = np.load(mf.with_name(mf.stem + "_linkcontact.npz"))["contact"]
        return torch.tensor(flags, device=env.device, dtype=torch.float32)

    return _cached(env, "_wuji_ref_link_contact", build)


def hand_link_contact(env, force_thresh: float = 1.0, near_ref: float = 0.05) -> torch.Tensor:
    """Contact fraction across 32 palm/finger links, with the same gate as fingertips.

    Reference flags come from <motion>_linkcontact.npz. This includes the
    shaft and palm contacts used by the SPIDER table-push reference."""
    mc = _get_motion_command_and_assert_type(env)
    idx = _hand_link_idx(env)
    t = mc.time_steps
    ref = _ref_link_contact(env)[t.clamp(max=_ref_link_contact(env).shape[0] - 1)]  # [N, 32]
    force = torch.norm(env.simulator.contact_forces[:, idx], dim=-1)  # [N, 32]
    p_ref = _tips_in_object(mc.motion.body_pos_w[t], mc.motion.object_pos_w[t], mc.motion.object_quat_w[t], idx)
    p_cur = _tips_in_object(env.simulator._rigid_body_pos, mc.simulator_object_pos_w, mc.simulator_object_quat_w, idx)
    on_object = torch.norm(p_cur - p_ref, dim=-1) < near_ref
    sim = ((force > force_thresh) & on_object).float()
    return (ref * sim).sum(-1) / ref.sum(-1).clamp(min=1.0)


def _abduction_dof_idx(env):
    """Eight non-thumb finger abduction joints: finger{2..5}_joint2 on each hand."""
    return _cached(
        env,
        "_wuji_abduction_dof_idx",
        lambda: torch.tensor(
            [
                i
                for i, n in enumerate(env.dof_names)
                if "finger" in n and n[-6:-1] == "joint" and n[-1] == "2" and "finger1" not in n
            ],
            device=env.device,
            dtype=torch.long,
        ),
    )


def finger_abduction_error_exp(env, sigma: float) -> torch.Tensor:
    """Track the eight non-thumb abduction angles; sigma is in radians."""
    mc = _get_motion_command_and_assert_type(env)
    idx = _abduction_dof_idx(env)
    err = torch.square(mc.joint_pos[:, idx] - env.simulator.dof_pos[:, idx]).mean(-1)
    return torch.exp(-err / sigma**2)


def object_displacement_error_exp(env, sigma: float) -> torch.Tensor:
    """Track object position with the historical displacement-term name.

    Both displacements subtract the reference start position, which cancels
    in their difference. This is a tighter absolute position kernel, not
    displacement from the actual simulated reset pose. Keep the expression
    and function name for compatibility with saved runs. Sigma is in metres."""
    mc = _get_motion_command_and_assert_type(env)
    t = mc.time_steps
    start = torch.minimum(torch.maximum(t - env.episode_length_buf, mc.motion.motion_start_idx[mc.motion_ids]), t)
    ref = mc.motion.object_pos_w  # [T, 3], no env origins
    d_ref = ref[t] - ref[start]
    d_sim = (mc.simulator_object_pos_w - env.simulator.scene.env_origins) - ref[start]
    err = torch.sum(torch.square(d_sim - d_ref), dim=-1)
    return torch.exp(-err / sigma**2)


def limits_dof_pos_body(env, soft_dof_pos_limit: float = 0.9) -> torch.Tensor:
    """Sum body-joint excursions outside soft limits; finger joints are excluded."""
    lim = env.simulator.hard_dof_pos_limits
    m = (lim[:, 0] + lim[:, 1]) / 2
    r = lim[:, 1] - lim[:, 0]
    lower, upper = m - 0.5 * r * soft_dof_pos_limit, m + 0.5 * r * soft_dof_pos_limit
    out = -(env.simulator.dof_pos - lower).clip(max=0.0) + (env.simulator.dof_pos - upper).clip(min=0.0)
    return torch.sum(out * _body_dof_mask(env), dim=1)
