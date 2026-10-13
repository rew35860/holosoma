"""HUMOTO training presets: reference, robot, rewards, resets and randomization.

Each registry sequence has base and v2-v7 presets named exp:wuji-<sequence>-vN.
Reward weights and sigmas are in configs/reward_wuji.py; formulas are in
rewards/terms.py. V2 scales reset noise, adjusts finger PD gains and relaxes
wrist/ankle z termination. V3-V7 vary rewards, preserving observation/action
conventions for checkpoint compatibility. V7 inherits from V5, not V6."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object
from wuji_wbt.configs.randomization_wuji import wuji_wbt_randomization_w_object
from wuji_wbt.configs.reward_wuji import (
    wuji_wbt_reward_w_object,
    wuji_wbt_reward_w_object_v3,
    wuji_wbt_reward_w_object_v4,
    wuji_wbt_reward_w_object_v5,
    wuji_wbt_reward_w_object_v6,
    wuji_wbt_reward_w_object_v7,
)
from wuji_wbt.configs.robot_wuji import g1_29dof_wuji_w_object
from wuji_wbt.configs.sequences import SEQUENCES, SequenceSpec

NUM_ITERATIONS = 20000  # Additional PPO iterations, including when resuming a checkpoint.
SAVE_INTERVAL = 1000

V2 = {
    "overall_noise_scale": 0.2,  # Reduce reset offsets for objects held in the hands.
    "start_at_timestep_zero_prob": 0.2,
    "finger_stiffness": 5.0,
    "finger_damping": 0.2,
    "bad_motion_body_pos_threshold": 0.35,  # Wrist/ankle z error, metres.
    "reward": wuji_wbt_reward_w_object,
    "tag": "-v2",
}

# All variants share V2's control/reset settings. Reward definitions contain their tuning deltas.
V3 = {**V2, "reward": wuji_wbt_reward_w_object_v3, "tag": "-v3"}
V4 = {**V3, "reward": wuji_wbt_reward_w_object_v4, "tag": "-v4"}
V5 = {**V4, "reward": wuji_wbt_reward_w_object_v5, "tag": "-v5"}
V6 = {**V5, "reward": wuji_wbt_reward_w_object_v6, "tag": "-v6"}
V7 = {**V5, "reward": wuji_wbt_reward_w_object_v7, "tag": "-v7"}


def make_experiment(seq: SequenceSpec, variant: dict | None = None):
    v = variant or {}
    base = g1_29dof_wbt_w_object
    motion_term = base.command.setup_terms["motion_command"]
    motion_cfg = motion_term.params["motion_config"]
    noise = motion_cfg.noise_to_initial_pose
    if v:
        noise = replace(noise, overall_noise_scale=v["overall_noise_scale"])
    motion_cfg = replace(
        motion_cfg,
        motion_file=str(seq.motion_npz),
        noise_to_initial_pose=noise,
        start_at_timestep_zero_prob=v.get("start_at_timestep_zero_prob", motion_cfg.start_at_timestep_zero_prob),
    )
    command = replace(
        base.command,
        setup_terms={
            **base.command.setup_terms,
            "motion_command": replace(motion_term, params={**motion_term.params, "motion_config": motion_cfg}),
        },
    )
    robot = g1_29dof_wuji_w_object
    control = robot.control
    if v:
        control = replace(
            control,
            stiffness={**control.stiffness, "finger": v["finger_stiffness"]},
            damping={**control.damping, "finger": v["finger_damping"]},
        )
    termination = base.termination
    if v:
        bt = termination.terms["bad_tracking"]
        termination = replace(
            termination,
            terms={
                **termination.terms,
                "bad_tracking": replace(
                    bt, params={**bt.params, "bad_motion_body_pos_threshold": v["bad_motion_body_pos_threshold"]}
                ),
            },
        )
    name = f"wuji_{seq.name}" + v.get("tag", "")
    randomization = wuji_wbt_randomization_w_object
    if seq.object_friction is not None:  # per-sequence object friction range (static; dynamic -0.1)
        lo, hi = seq.object_friction
        term = randomization.setup_terms["randomize_object_rigid_body_material_startup"]
        randomization = replace(
            randomization,
            setup_terms={
                **randomization.setup_terms,
                "randomize_object_rigid_body_material_startup": replace(
                    term,
                    params={
                        **term.params,
                        "static_friction_range": [lo, hi],
                        "dynamic_friction_range": [round(lo - 0.1, 3), round(hi - 0.1, 3)],
                    },
                ),
            },
        )
    return replace(
        base,
        training=replace(base.training, name=name),
        termination=termination,
        algo=replace(
            base.algo,
            config=replace(base.algo.config, num_learning_iterations=NUM_ITERATIONS, save_interval=SAVE_INTERVAL),
        ),
        robot=replace(
            robot,
            control=control,
            asset=replace(robot.asset, enable_self_collisions=True),
            object=replace(robot.object, object_urdf_path=seq.object_urdf),
            init_state=replace(robot.init_state, pos=[0.0, 0.0, 0.76]),
        ),
        command=command,
        reward=v.get("reward", wuji_wbt_reward_w_object),
        randomization=randomization,
    )


VARIANTS = {"": None, "-v2": V2, "-v3": V3, "-v4": V4, "-v5": V5, "-v6": V6, "-v7": V7}
PRESETS = {
    f"exp:wuji-{name}{suffix}": make_experiment(seq, variant)
    for suffix, variant in VARIANTS.items()
    for name, seq in SEQUENCES.items()
}
