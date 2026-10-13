"""HUMOTO reward weights and kernel widths for the original Wuji hands.

Edit RewardTermCfg.weight for relative importance and params["sigma"] for
tracking tolerance (metres or radians). The reward manager applies control dt.
Body/object defaults come from holosoma.config_values.wbt.g1.reward.
Formulas live in wuji_wbt.rewards.terms; re-exports below preserve the function
paths stored in existing checkpoints and configuration files.

CLI example, after exp:wuji-humoto-648-v7:
    --reward.terms.fingertip-pos-object-error-exp.weight=6.0
    --reward.terms.fingertip-pos-object-error-exp.params.sigma=0.10
"""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object
from wuji_wbt.rewards.terms import (
    HAND_LINKS,
    PALMS,
    TIPS,
    finger_abduction_error_exp,
    finger_joint_pos_error_exp,
    fingertip_contact,
    fingertip_pos_object_error_exp,
    fingertip_pos_palm_error_exp,
    hand_link_contact,
    limits_dof_pos_body,
    object_displacement_error_exp,
)

UNDESIRED_CONTACTS_REGEX = (
    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
    "(?!wj_)(?!wjr_).+$"
)

# Base/v2: body + object tracking, finger angles, and hand shape.
wuji_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object.terms,
        "limits_dof_pos": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:limits_dof_pos_body",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={"threshold": 1.0, "undesired_contacts_body_names": UNDESIRED_CONTACTS_REGEX},
            weight=-0.1,
        ),
        # hand tracker
        "finger_joint_pos_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:finger_joint_pos_error_exp",
            params={"sigma": 0.35},  # rad: a 20 deg RMS finger error -> exp(-1)
            weight=1.0,
        ),
        "fingertip_pos_palm_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:fingertip_pos_palm_error_exp",
            params={"sigma": 0.03},  # m: a 3 cm RMS fingertip error -> exp(-1)
            weight=1.0,
        ),
    }
)

# v3: add fingertip positions in the object frame.
wuji_wbt_reward_w_object_v3 = RewardManagerCfg(
    terms={
        **wuji_wbt_reward_w_object.terms,
        "fingertip_pos_object_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:fingertip_pos_object_error_exp",
            params={"sigma": 0.05},  # m: a 5 cm RMS fingertip-on-object error -> exp(-1)
            weight=2.0,
        ),
    }
)

# v4: contact force, abduction, and a tighter object-position kernel.
wuji_wbt_reward_w_object_v4 = RewardManagerCfg(
    terms={
        **wuji_wbt_reward_w_object_v3.terms,
        "fingertip_pos_object_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:fingertip_pos_object_error_exp",
            params={"sigma": 0.08},
            weight=4.0,
        ),
        "fingertip_contact": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:fingertip_contact",
            params={"force_thresh": 1.0},  # N of net contact force on a fingertip link
            weight=2.0,
        ),
        "finger_abduction_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:finger_abduction_error_exp",
            params={"sigma": 0.1},  # rad: a 5.7 deg RMS abduction error -> exp(-1)
            weight=1.0,
        ),
        "object_displacement_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:object_displacement_error_exp",
            params={"sigma": 0.05},  # m: a 5 cm object-position error -> exp(-1)
            weight=2.0,
        ),
    }
)

# v5: stronger contact/abduction terms; object sigma widened to 8 cm.
wuji_wbt_reward_w_object_v5 = RewardManagerCfg(
    terms={
        **wuji_wbt_reward_w_object_v4.terms,
        "fingertip_contact": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:fingertip_contact",
            params={"force_thresh": 1.0},
            weight=4.0,
        ),
        "finger_abduction_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:finger_abduction_error_exp",
            params={"sigma": 0.1},
            weight=3.0,
        ),
        "object_displacement_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:object_displacement_error_exp",
            params={"sigma": 0.08},
            weight=2.0,
        ),
    }
)

# v6: all-hand-link contact replaces tip contact; stronger finger tracking.
wuji_wbt_reward_w_object_v6 = RewardManagerCfg(
    terms={
        **{k: v for k, v in wuji_wbt_reward_w_object_v5.terms.items() if k != "fingertip_contact"},
        "hand_link_contact": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:hand_link_contact",
            params={"force_thresh": 1.0, "near_ref": 0.05},
            weight=4.0,
        ),
        "finger_joint_pos_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:finger_joint_pos_error_exp",
            params={"sigma": 0.15},  # rad: an 8.6 deg RMS finger error -> exp(-1)
            weight=3.0,
        ),
    }
)

# v7 derives from v5: tip contact and a wider abduction sigma (0.2 rad).
wuji_wbt_reward_w_object_v7 = RewardManagerCfg(
    terms={
        **wuji_wbt_reward_w_object_v5.terms,
        "finger_abduction_error_exp": RewardTermCfg(
            func="wuji_wbt.configs.reward_wuji:finger_abduction_error_exp",
            params={"sigma": 0.2},
            weight=3.0,
        ),
    }
)


__all__ = [
    "HAND_LINKS",
    "PALMS",
    "TIPS",
    "UNDESIRED_CONTACTS_REGEX",
    "finger_abduction_error_exp",
    "finger_joint_pos_error_exp",
    "fingertip_contact",
    "fingertip_pos_object_error_exp",
    "fingertip_pos_palm_error_exp",
    "hand_link_contact",
    "limits_dof_pos_body",
    "object_displacement_error_exp",
    "wuji_wbt_reward_w_object",
    "wuji_wbt_reward_w_object_v3",
    "wuji_wbt_reward_w_object_v4",
    "wuji_wbt_reward_w_object_v5",
    "wuji_wbt_reward_w_object_v6",
    "wuji_wbt_reward_w_object_v7",
]
