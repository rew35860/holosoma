"""CPU checks for HUMOTO reward semantics and checkpoint compatibility."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from wuji_wbt.configs import reward_wuji
from wuji_wbt.configs.experiment_wuji import PRESETS
from wuji_wbt.configs.robot_wuji import g1_29dof_wuji
from wuji_wbt.rewards import terms


@pytest.fixture
def reward_env(monkeypatch):
    num_envs, frames = 3, 4
    names = terms.HAND_LINKS
    bodies, joints = len(names), len(g1_29dof_wuji.dof_names)
    body_quat = torch.zeros(frames, bodies, 4)
    body_quat[..., 3] = 1.0
    object_quat = torch.zeros(frames, 4)
    object_quat[:, 3] = 1.0
    motion = SimpleNamespace(
        body_pos_w=torch.zeros(frames, bodies, 3),
        body_quat_w=body_quat,
        object_pos_w=torch.zeros(frames, 3),
        object_quat_w=object_quat,
        motion_start_idx=torch.tensor([0]),
    )
    time_steps = torch.arange(num_envs)
    command = SimpleNamespace(
        motion=motion,
        time_steps=time_steps,
        motion_ids=torch.zeros(num_envs, dtype=torch.long),
        joint_pos=torch.zeros(num_envs, joints),
        simulator_object_pos_w=motion.object_pos_w[time_steps].clone(),
        simulator_object_quat_w=motion.object_quat_w[time_steps].clone(),
    )
    sim = SimpleNamespace(
        _body_list=names,
        dof_pos=command.joint_pos.clone(),
        _rigid_body_pos=motion.body_pos_w[time_steps].clone(),
        _rigid_body_rot=motion.body_quat_w[time_steps].clone(),
        contact_forces=torch.zeros(num_envs, bodies, 3),
        hard_dof_pos_limits=torch.tensor([[-1.0, 1.0]]).repeat(joints, 1),
        scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
    )
    env = SimpleNamespace(
        device="cpu",
        dof_names=g1_29dof_wuji.dof_names,
        simulator=sim,
        command=command,
        episode_length_buf=time_steps.clone(),
        _wuji_ref_tip_contact=torch.ones(frames, 10),
        _wuji_ref_link_contact=torch.ones(frames, bodies),
    )
    monkeypatch.setattr(terms, "_get_motion_command_and_assert_type", lambda _: command)
    return env


@pytest.mark.parametrize(
    "function",
    [
        terms.finger_joint_pos_error_exp,
        terms.finger_abduction_error_exp,
        terms.fingertip_pos_palm_error_exp,
        terms.fingertip_pos_object_error_exp,
        terms.object_displacement_error_exp,
    ],
)
def test_exact_reference_has_unit_tracking_reward(reward_env, function):
    torch.testing.assert_close(function(reward_env, sigma=0.1), torch.ones(3))


def test_finger_kernel_uses_mean_and_excludes_body_joints(reward_env):
    sigma = 0.35
    reward_env.simulator.dof_pos[:] = 100.0
    idx = terms._finger_dof_idx(reward_env)
    reward_env.simulator.dof_pos[:, idx] = sigma
    torch.testing.assert_close(
        terms.finger_joint_pos_error_exp(reward_env, sigma),
        torch.exp(torch.full((3,), -1.0)),
    )


def test_body_soft_limits_exclude_fingers(reward_env):
    reward_env.simulator.dof_pos[:, terms._finger_dof_idx(reward_env)] = 10.0
    torch.testing.assert_close(terms.limits_dof_pos_body(reward_env), torch.zeros(3))
    reward_env.simulator.dof_pos[:, 0] = 1.0
    torch.testing.assert_close(terms.limits_dof_pos_body(reward_env), torch.full((3,), 0.1))


@pytest.mark.parametrize(
    ("function", "cache", "index"),
    [
        (terms.fingertip_contact, "_wuji_ref_tip_contact", lambda env: terms._hand_body_idx(env)[1].flatten()),
        (terms.hand_link_contact, "_wuji_ref_link_contact", terms._hand_link_idx),
    ],
)
def test_contact_requires_force_proximity_and_reference(reward_env, function, cache, index):
    env = reward_env
    ids = index(env)
    torch.testing.assert_close(function(env), torch.zeros(3))
    env.simulator.contact_forces[:, ids, 0] = 2.0
    torch.testing.assert_close(function(env), torch.ones(3))
    # Force on a link far from the reference must not count.
    env.simulator._rigid_body_pos[:, ids, 0] = 0.2
    torch.testing.assert_close(function(env), torch.zeros(3))
    env.simulator._rigid_body_pos.zero_()
    getattr(env, cache).zero_()
    torch.testing.assert_close(function(env), torch.zeros(3))


def test_object_frame_tracking_is_invariant_to_world_translation(reward_env):
    env = reward_env
    generator = torch.Generator().manual_seed(7)
    env.command.motion.body_pos_w[:] = torch.rand(env.command.motion.body_pos_w.shape, generator=generator)
    env.simulator._rigid_body_pos[:] = env.command.motion.body_pos_w[env.command.time_steps] + 0.01
    before = terms.fingertip_pos_object_error_exp(env, sigma=0.08)
    translation = torch.tensor([1.5, -2.0, 0.7])
    env.command.motion.body_pos_w += translation
    env.command.motion.object_pos_w += translation
    env.simulator._rigid_body_pos += translation
    env.command.simulator_object_pos_w += translation
    torch.testing.assert_close(terms.fingertip_pos_object_error_exp(env, sigma=0.08), before)


def test_historical_displacement_term_is_tighter_position_tracking(reward_env):
    env = reward_env
    env.command.motion.object_pos_w[:, 0] = torch.tensor([0.3, 0.4, 0.6, 0.8])
    env.simulator.scene.env_origins[:, 0] = torch.tensor([0.0, 3.0, 6.0])
    env.command.simulator_object_pos_w[:] = (
        env.command.motion.object_pos_w[env.command.time_steps] + env.simulator.scene.env_origins
    )
    env.command.simulator_object_pos_w[:, 2] += 0.08
    torch.testing.assert_close(
        terms.object_displacement_error_exp(env, sigma=0.08),
        torch.exp(torch.full((3,), -1.0)),
    )


def test_saved_reward_function_paths_still_resolve():
    for preset in PRESETS.values():
        for term in preset.reward.terms.values():
            module, name = term.func.split(":")
            function = getattr(importlib.import_module(module), name)
            assert callable(function)
            if module == "wuji_wbt.configs.reward_wuji":
                assert function is getattr(terms, name)


def test_v7_keeps_v5_tip_contact_and_v6_uses_hand_links():
    v6 = reward_wuji.wuji_wbt_reward_w_object_v6.terms
    v7 = reward_wuji.wuji_wbt_reward_w_object_v7.terms
    assert "hand_link_contact" in v6 and "fingertip_contact" not in v6
    assert "fingertip_contact" in v7 and "hand_link_contact" not in v7
    assert v6["finger_joint_pos_error_exp"].params["sigma"] == 0.15
    assert v7["finger_joint_pos_error_exp"].params["sigma"] == 0.35
    assert v7["finger_abduction_error_exp"].params["sigma"] == 0.2


def test_existing_training_cli_accepts_reward_overrides(monkeypatch):
    from wuji_wbt.launch import train_wuji

    received = []
    monkeypatch.setattr(train_wuji, "train", received.append)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_wuji.py",
            "exp:wuji-humoto-648-v7",
            "logger:disabled",
            "--reward.terms.fingertip-pos-object-error-exp.weight=6.0",
            "--reward.terms.fingertip-pos-object-error-exp.params.sigma=0.10",
        ],
    )
    try:
        train_wuji.main()
    finally:
        # The launcher patches Holosoma at import time; leave other tests unaffected.
        train_wuji.TA.init_sim_imports = train_wuji._orig_init
    term = received[0].reward.terms["fingertip_pos_object_error_exp"]
    assert term.weight == 6.0
    assert term.params["sigma"] == 0.10
