"""g1_29dof_wuji RobotConfig (69 DOF = 29 body + 40 Wuji fingers).

Reuses holosoma's `g1_29dof` verbatim via dataclasses.replace and APPENDS the 40 finger joints
(names + per-DOF limits/effort/velocity from the Wuji URDF, plus chosen finger PD gains / armature).
holosoma is DOF-agnostic: action_dim, PD gains (substring-matched) and the PPO net all size from this
config + the URDF at runtime, so a 69-DOF robot needs no holosoma edits.

Asset: the G1+Wuji URDF was relocated into holosoma/data/robots/g1_wuji/ (asset copy, not a code change).
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_values.robot import g1_29dof, g1_29dof_w_object

# --- 40 Wuji finger joints, in the URDF's DOF order (left 20 then right 20) ---
WUJI_FINGER_JOINTS = [f"wj_left_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)] + [
    f"wjr_right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)
]

# per-DOF limits read from models/g1_wuji/g1_29dof_wuji.urdf (left 20 pattern repeated for right 20)
_L20_LOWER = [
    0.0475,
    -0.1387,
    -0.4642,
    -0.4699,
    -0.1585,
    -0.37,
    -0.4777,
    -0.4683,
    -0.1644,
    -0.37,
    -0.4739,
    -0.4684,
    -0.1554,
    -0.37,
    -0.4765,
    -0.4777,
    -0.1626,
    -0.37,
    -0.4768,
    -0.4683,
]
_L20_UPPER = [
    1.6033,
    0.9324,
    1.5623,
    1.5568,
    1.5604,
    0.37,
    1.5485,
    1.5753,
    1.5516,
    0.37,
    1.5512,
    1.5745,
    1.5585,
    0.37,
    1.5487,
    1.5634,
    1.5585,
    0.37,
    1.549,
    1.5735,
]
_L20_VEL = [
    8.587,
    11.1,
    12.863,
    13.578,
    8.203,
    8.115,
    12.863,
    13.578,
    8.203,
    8.115,
    12.863,
    13.578,
    8.203,
    8.115,
    12.863,
    13.578,
    8.203,
    8.115,
    12.863,
    13.578,
]
_L20_EFFORT = [
    0.4452,
    0.4259,
    0.1888,
    0.1468,
    0.6188,
    0.1822,
    0.2251,
    0.217,
    0.6494,
    0.1827,
    0.2078,
    0.2018,
    0.6389,
    0.1832,
    0.2249,
    0.2044,
    0.6441,
    0.1798,
    0.2384,
    0.1866,
]
FINGER_LOWER = _L20_LOWER * 2
FINGER_UPPER = _L20_UPPER * 2
FINGER_VEL = _L20_VEL * 2
FINGER_EFFORT = _L20_EFFORT * 2
FINGER_ARMATURE = [0.001] * 40  # small; URDF has none -> chosen (light, low-inertia fingers)
FINGER_FRICTION = [0.0] * 40
# neutral default pose CLAMPED into each joint's limits (finger1_joint1 lower=0.0475 excludes 0.0)
FINGER_DEFAULTS = [min(max(0.0, lo), hi) for lo, hi in zip(FINGER_LOWER, FINGER_UPPER)]

# Native tip links; Isaac Sim merges these fixed links into the distal finger links.
WUJI_FINGERTIP_LINKS = [f"wj_left_finger{f}_tip_link" for f in range(1, 6)] + [
    f"wjr_right_finger{f}_tip_link" for f in range(1, 6)
]

# Body names after Isaac Sim merges fixed links. Keep these aligned with the imported URDF.
WUJI_BODY_NAMES = [
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "waist_support_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "wj_left_palm_link",
    "wj_left_finger1_link1",
    "wj_left_finger1_link2",
    "wj_left_finger1_link3",
    "wj_left_finger1_link4",
    "wj_left_finger2_link1",
    "wj_left_finger2_link2",
    "wj_left_finger2_link3",
    "wj_left_finger2_link4",
    "wj_left_finger3_link1",
    "wj_left_finger3_link2",
    "wj_left_finger3_link3",
    "wj_left_finger3_link4",
    "wj_left_finger4_link1",
    "wj_left_finger4_link2",
    "wj_left_finger4_link3",
    "wj_left_finger4_link4",
    "wj_left_finger5_link1",
    "wj_left_finger5_link2",
    "wj_left_finger5_link3",
    "wj_left_finger5_link4",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "wjr_right_palm_link",
    "wjr_right_finger1_link1",
    "wjr_right_finger1_link2",
    "wjr_right_finger1_link3",
    "wjr_right_finger1_link4",
    "wjr_right_finger2_link1",
    "wjr_right_finger2_link2",
    "wjr_right_finger2_link3",
    "wjr_right_finger2_link4",
    "wjr_right_finger3_link1",
    "wjr_right_finger3_link2",
    "wjr_right_finger3_link3",
    "wjr_right_finger3_link4",
    "wjr_right_finger4_link1",
    "wjr_right_finger4_link2",
    "wjr_right_finger4_link3",
    "wjr_right_finger4_link4",
    "wjr_right_finger5_link1",
    "wjr_right_finger5_link2",
    "wjr_right_finger5_link3",
    "wjr_right_finger5_link4",
]

_FINGER_STIFFNESS = 2.0  # substring key "finger" matches all 40 (no body joint contains it)
_FINGER_DAMPING = 0.1

g1_29dof_wuji = replace(
    g1_29dof,
    actions_dim=69,
    dof_obs_size=69,
    num_bodies=len(WUJI_BODY_NAMES),
    body_names=WUJI_BODY_NAMES,
    key_bodies=["left_ankle_roll_link", "right_ankle_roll_link"],  # no foot_contact_point in this URDF
    foot_height_name="ankle_roll_link",
    dof_names=[*g1_29dof.dof_names, *WUJI_FINGER_JOINTS],
    dof_pos_lower_limit_list=[*g1_29dof.dof_pos_lower_limit_list, *FINGER_LOWER],
    dof_pos_upper_limit_list=[*g1_29dof.dof_pos_upper_limit_list, *FINGER_UPPER],
    dof_vel_limit_list=[*g1_29dof.dof_vel_limit_list, *FINGER_VEL],
    dof_effort_limit_list=[*g1_29dof.dof_effort_limit_list, *FINGER_EFFORT],
    dof_armature_list=[*g1_29dof.dof_armature_list, *FINGER_ARMATURE],
    dof_joint_friction_list=[*g1_29dof.dof_joint_friction_list, *FINGER_FRICTION],
    init_state=replace(
        g1_29dof.init_state,
        default_joint_angles={
            **g1_29dof.init_state.default_joint_angles,
            **{j: d for j, d in zip(WUJI_FINGER_JOINTS, FINGER_DEFAULTS)},
        },
    ),
    control=replace(
        g1_29dof.control,
        stiffness={**g1_29dof.control.stiffness, "finger": _FINGER_STIFFNESS},
        damping={**g1_29dof.control.damping, "finger": _FINGER_DAMPING},
    ),
    asset=replace(
        g1_29dof.asset,
        urdf_file="g1_wuji/g1_29dof_wuji.urdf",
        robot_type="g1_29dof_wuji",
    ),
)

# object variant: reuse g1_29dof_w_object's object sub-config (object_urdf_path set in the exp preset)
g1_29dof_wuji_w_object = replace(g1_29dof_wuji, object=g1_29dof_w_object.object)
