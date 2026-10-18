#!/usr/bin/env python
"""Scale-consistent whole-body G1+Wuji retargeting, with explicit v1/v2 selection.

Dataset-agnostic (humoto now; other InterAct-format sources later). One uniformly-scaled object is used for
source, target, collision, render and the future RL env. Stages:
  PART 0  build ONE canonical FLOOR-ANCHORED UNIFORM reference (+ one scaled table model).
  PART 1  OmniRetarget Stage-A whole body, scaled object as BOTH source and target, robot-table collision ON.
  V1 PART 2  independently floating TopoRetarget hands.
  V1 PART 3  arm/finger refinement, then contact-blended arms + floating fingers.
  V2 PART 2  quick finger initialization, coupled arms/fingers, direct 30/50 Hz export.
  PART 4  reserved; rendering uses the separate comparison renderer.

Uniform floor-anchored transform (NO floor-rest z0 branch, NO double scaling):
    c = [0,0,z_min];  X_scaled = s*(X_raw - c);  quat unchanged;  object_local_scaled = s*object_local_raw

Run (omniretarget env):
    python retarget_pipeline/pipeline.py --task-name push_table_forward_with_both_hands-039 --stage part0 --data-path <dir>
    python retarget_pipeline/pipeline.py --task-name <task> --stage all --data-path <dir>
"""

from __future__ import annotations
import sys, logging
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import tyro

PKG = Path(__file__).resolve().parents[1]                        # package root .../holosoma_retargeting/ (for models/, outputs)
src_root = PKG.parent                                            # dir CONTAINING the package (for `holosoma_retargeting.*` imports)
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))                           # local source wins over the pip-installed copy

from holosoma_retargeting.wuji.topoRetarget import TopoRetargeter
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.config_types.robot import _ROBOT_DEFAULTS, RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import RetargeterConfig, SelfCollisionConfig
from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.retarget_pipeline.config_types.pipeline_config import PipelineConfig  # noqa: E402
from holosoma_retargeting.retarget_pipeline import checks, report  # noqa: E402  (fail-loud checks + console reports)
from holosoma_retargeting.examples.robot_retarget import (  # lazy: keep PART 0 free of the retargeter deps
    create_task_constants,
    build_retargeter_kwargs_from_config,
    initialize_robot_pose)

from holosoma_retargeting.src.utils import (
    load_intermimic_data,
    calculate_scale_factor,
    load_object_data,  # noqa: E402
    extract_foot_sticking_sequence_velocity
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_config(cfg: PipelineConfig) -> None:
    """Validate configuration consistency.

    Args:
        cfg: Pipeline configuration.

    Raises:
        ValueError: If configuration is invalid.
    """
    if cfg.data_path is None:
        raise ValueError("data_path is required (raw dataset lives outside the project); pass --data-path <path>")

    if cfg.robot not in _ROBOT_DEFAULTS:
        raise ValueError(f"Unknown robot '{cfg.robot}'. Available: {sorted(_ROBOT_DEFAULTS)}")

    pt = cfg.data_path / f"{cfg.task_name}.pt"
    handkp = cfg.data_path / f"{cfg.task_name}_handkp.npy"

    if not pt.exists():
        raise ValueError(f"Raw motion file not found: {pt}")
    if not handkp.exists():
        logger.warning("no hand keypoints (%s) -- PART 0/1 run body-only; PART 2/3 unavailable", handkp)


def preprocess_motion_data_uniform(human_joints, hand_kp, object_poses, demo_joints, foot_names, scale, mat_height=0.1):
    """Normalize joint heights to the floor, then uniformly scale x, y and z: X_scaled = s*(X - [0,0,z_min]).

    Args:
        human_joints (np.ndarray): Human joint positions (T, 52, 3).
        hand_kp (np.ndarray): Hand keypoints (T, 2, 21, 3).
        object_poses (np.ndarray): Object poses (T, 7) [qw, qx, qy, qz, x, y, z].
        demo_joints (list): Ordered joint names (resolves foot_names -> indices).
        foot_names (tuple): The two toe joint names defining the floor z_min.
        scale (float): Uniform scale factor s.
        mat_height (float): Mat thickness subtracted from z_min when the feet rest on a mat.

    Returns:
        tuple: (hj_s, hk_s, obj_pose_muj, z_min).
    """
    # Normalize human joint heights
    toe_indices = [demo_joints.index(foot_names[0]), demo_joints.index(foot_names[1])]
    z_min = float(human_joints[:, toe_indices, 2].min())
    if z_min >= mat_height:
        # On a mat.
        z_min -= mat_height

    c = np.array([0.0, 0.0, z_min])

    # Uniformly scale human joints, hand keypoints and object translation about the floor
    hj_s = scale * (human_joints - c)
    hk_s = scale * (hand_kp - c) if hand_kp is not None else None    # None when the dataset has no hand keypoints
    obj_pos_s = scale * (object_poses[:, 4:7] - c)                   # translation scaled; quaternion unchanged
    obj_pose_muj = np.concatenate([obj_pos_s, object_poses[:, :4]], axis=1)  # -> [x,y,z, qw,qx,qy,qz] MuJoCo order
    return hj_s, hk_s, obj_pose_muj, z_min


def load_raw_motion(pt, handkp):
    """Load raw human joints, hand keypoints and object poses; truncate to the common frame count.

    Args:
        pt (str): Path to the <task_name>.pt file (human joints + object poses).
        handkp (str): Path to the <task_name>_handkp.npy file (hand keypoints).

    Returns:
        tuple: (raw_hj, raw_hk, raw_obj), each truncated to T frames.
    """
    raw_hj, raw_obj = load_intermimic_data(pt)                       # (T,52,3) ; (T,7)[qw,qx,qy,qz,x,y,z]
    raw_hk = np.load(handkp) if Path(handkp).exists() else None      # (T,2,21,3) or None (e.g. OMOMO: no hand kp)
    lens = [len(raw_hj), len(raw_obj)] + ([len(raw_hk)] if raw_hk is not None else [])
    T = min(lens)
    return raw_hj[:T], (raw_hk[:T] if raw_hk is not None else None), raw_obj[:T]


def determine_output_path(cfg: PipelineConfig, part: str) -> str:
    """Determine the output file path for a pipeline part.

    Args:
        cfg: Pipeline configuration.
        part: Pipeline stage ("part0", "part1", "part2", "part3", "part4").

    Returns:
        Output file path for that part (parent directory created).
    """
    if part == "part4":                                   # render -> renders/part4/ (outside the data save_dir)
        path = PKG / "renders" / "part4" / f"{cfg.task_name}.mp4"
    else:
        names = {
            "part0": f"part0/{cfg.task_name}_canonical.npz",  # canonical scaled reference
            "part1": f"part1/{cfg.task_name}_stageA.npz",     # OmniRetarget whole-body qpos
            "part2": f"part2/{cfg.task_name}_stageB.npz",     # attached TopoRetarget hands
            "part3": f"part3/{cfg.task_name}_final.npz",      # assembled final qpos
        }
        if part not in names:
            raise ValueError(f"Unknown part: {part}")
        path = PKG / cfg.save_dir / names[part]
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def require_exists(paths, hint: str) -> None:
    """Fail loud if any prerequisite input path is missing.

    Args:
        paths (iterable): Paths that must already exist.
        hint (str): How to produce them (shown in the error).
    """
    for p in paths:
        if not Path(p).exists():
            raise ValueError(f"required input missing: {p}\n  {hint}")


def resolve_model_base(cfg: PipelineConfig, stage: str) -> Path:
    """Resolve a stage's base (unscaled) MuJoCo model path.

    Uses the config override when given, else auto-resolves by robot + object (+ cfg.stageA_hand for Stage A):
        Stage A, wuji   -> models/<robot>/<robot>_29dof_wuji_welded_w_<object_name>.xml  (Wuji fingers WELDED)
        Stage A, rubber -> models/<robot>/<robot>_29dof_w_<object_name>.xml              (rubber mitten)
        Stage B         -> models/<robot>/hand_both_w_<object_name>_coacd_ball.xml       (floating hands + CoACD)

    Args:
        cfg (PipelineConfig): Pipeline configuration.
        stage (str): "A" or "B".

    Returns:
        Path: Absolute path to the base model (existence NOT checked here -- callers that load it check).
    """
    override = cfg.stageA_model_base if stage == "A" else cfg.stageB_model_base
    if override is not None:
        return PKG / override
    if stage == "A":
        stem = (f"{cfg.robot}_29dof_wuji_welded_w_{cfg.object_name}" if cfg.stageA_hand == "wuji"
                else f"{cfg.robot}_29dof_w_{cfg.object_name}")
    else:
        stem = f"hand_both_w_{cfg.object_name}_coacd_ball"
    return PKG / "models" / cfg.robot / f"{stem}.xml"


def build_scaled_object_model(base_model_path: str, s: float, out_path: str, obj_name: str) -> tuple[str, int]:
    """Uniformly scale every object mesh (visible + CoACD pieces) in a MuJoCo model by s, write out_path.

    Object meshes follow the `<obj_name>_...` convention (visible `<obj_name>_mesh`, collision
    `<obj_name>_piece_NNN_mesh`). The visible surface and its CoACD pieces derive from the same source .obj,
    so scaling both by the same s keeps them flush (no collision under-fill). `meshdir` is rewritten absolute
    so the scaled model resolves its assets from anywhere (it lives with the run artifacts, not beside the base).

    Args:
        base_model_path (str): Base (unscaled) MuJoCo XML.
        s (float): Uniform scale factor.
        out_path (str): Destination XML path.
        obj_name (str): Interaction object name (prefix of the object mesh names to scale).

    Returns:
        tuple: (out_path, number of object meshes scaled).
    """
    base_dir = Path(base_model_path).resolve().parent
    tree = ET.parse(base_model_path)
    root = tree.getroot()
    comp = root.find("compiler")                                     # make meshdir absolute -> model is relocatable
    if comp is not None:
        md = comp.get("meshdir")
        if md and not Path(md).is_absolute():
            comp.set("meshdir", str((base_dir / md).resolve()))

    n = 0
    for mesh in root.iter("mesh"):
        if (mesh.get("name") or "").startswith(obj_name):           # visible <obj>_mesh + every <obj>_piece_NNN_mesh
            mesh.set("scale", f"{s} {s} {s}")
            n += 1

    if n == 0:
        raise ValueError(f"no '{obj_name}' meshes in {base_model_path} -- nothing to scale (fail loud)")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path)

    return out_path, n


def scaled_model_path(cfg: PipelineConfig, stage: str) -> Path:
    """Deterministic path of a stage's SCALED model: models/scaled/stage<X>/<base name>."""
    return PKG / cfg.save_dir / "models" / cfg.task_name / f"stage{stage}" / resolve_model_base(cfg, stage).name


def build_scaled_model(cfg: PipelineConfig, stage: str, s: float) -> tuple[str, str, int]:
    """Resolve + fail-loud-check a stage's base model, scale its object by s into models/scaled/stage<X>/.

    Args:
        cfg (PipelineConfig): Pipeline configuration.
        stage (str): "A" or "B".
        s (float): Uniform scale factor.
    Returns:
        tuple: (base_path, scaled_path, n_meshes).
    """
    base = resolve_model_base(cfg, stage)
    if not base.exists():
        raise ValueError(f"Stage-{stage} base model not found: {base}\n"
                         f"  build the '{cfg.object_name}' base (visible mesh + CoACD pieces), "
                         f"or pass --stage-{stage.lower()}-model-base <path>.")
    scaled, n = build_scaled_object_model(str(base), s, str(scaled_model_path(cfg, stage)), cfg.object_name)
    return str(base), scaled, n


# ============================ PART 0 : canonical scaled reference ============================

def part0(cfg: PipelineConfig, motion_cfg: MotionDataConfig, con: SimpleNamespace) -> tuple[str, str]:
    """PART 0: canonical floor-anchored reference (.npz) + scaled Stage-A object model (.xml).

    One uniform scale s (and one set of task constants) drives BOTH outputs. Fail-loud checks throughout.
    Args:
        cfg (PipelineConfig): Pipeline configuration.
    Returns:
        tuple: (canonical reference .npz path, scaled Stage-A model .xml path).
    """
    task_name = cfg.task_name
    pt = str(cfg.data_path / f"{task_name}.pt")
    handkp = str(cfg.data_path / f"{task_name}_handkp.npy")

    # --- task constants + the single uniform scale s (shared by BOTH outputs below) ---
    s = float(calculate_scale_factor(task_name, con.ROBOT_HEIGHT))
    human_height = checks.assert_single_scale(s, con.ROBOT_HEIGHT)      # no-double-scale

    # --- (a) canonical floor-anchored reference (.npz) ---
    raw_hj, raw_hk, raw_obj = load_raw_motion(pt, handkp)
    T = len(raw_hj)

    # normalize heights to the floor + uniform scale (mat-aware z_min from foot_names, like preprocess_motion_data)
    hj_s, hk_s, obj_pose_muj, z_min = preprocess_motion_data_uniform(
        human_joints=raw_hj,
        hand_kp=raw_hk,
        object_poses=raw_obj,
        demo_joints=con.DEMO_JOINTS,
        foot_names=motion_cfg.toe_names,
        scale=s
    )

    stand = float(raw_hj[:, :, 2].max() - z_min)

    # object surface points: returns (raw, scaled)  [points_scaled = points * s]
    obj_local_raw, obj_local_s = load_object_data(f"{PKG}/{con.OBJECT_MESH_FILE}", smpl_scale=s, sample_count=cfg.n_obj)
    checks.assert_object_local_scaled(obj_local_s, obj_local_raw, s)
    off = None if raw_hk is None else checks.assert_contact_offsets_preserved(raw_hk, hk_s, raw_obj, obj_pose_muj, s)

    ref = determine_output_path(cfg, "part0")
    np.savez(ref, human_joints_scaled=hj_s, hand_kp_scaled=(hk_s if hk_s is not None else np.array([])),
             object_pose_mujoco=obj_pose_muj,
             object_points_scaled=obj_local_s, scale=s, z_min=z_min, frame_ids=np.arange(T), robot=cfg.robot,
             dataset=cfg.dataset, input_pt=pt, input_handkp=handkp, obj_mesh=con.OBJECT_MESH_FILE,
             stageA_model_base=str(resolve_model_base(cfg, "A")), stageB_model_base=str(resolve_model_base(cfg, "B")),
             transform="X_scaled = s*(X_raw - [0,0,z_min]); quat unchanged; UNIFORM floor-anchored")

    report.part0(cfg, task_name, T, s, z_min, human_height, stand, off, ref)

    # --- (b) scaled Stage-A object model (visible + CoACD, SAME s) ---
    base_a, scaled_a, n_a = build_scaled_model(cfg, "A", s)
    geo = checks.assert_object_model_scaled(base_a, scaled_a, s, cfg.object_name)   # scale applied + no new under-fill
    report.part0_model(cfg, scaled_a, n_a, s, geo)

    return ref, scaled_a


# ============================ PART 1 : OmniRetarget Stage-A whole body ============================

# G1 arm<->trunk self-collision pairs. ONLY the elbow vs the upper trunk: the elbow is the link that swings
# through the torso, and it is the pair that stays feasible. Wrist<->waist/pelvis pairs are excluded -- the
# wrists sit naturally within a few cm of the waist/pelvis during carries, so a hard clearance there makes the
# QP infeasible at frame 0. Enabled by --self-collision (tolerance below is a small non-overlap margin).
_ARM_TRUNK_PAIRS = [(f"{side}_elbow_link", trunk)
                    for side in ("left", "right")
                    for trunk in ("torso_link", "waist_support_link")]


def part1_stageA(cfg: PipelineConfig, motion_cfg: MotionDataConfig, con: SimpleNamespace) -> str:
    """PART 1: OmniRetarget Stage-A whole body (body + wrist) against the SCALED Stage-A model.

    Consumes PART 0's canonical .npz and mirrors robot_retarget.main, with ONE change: the retargeter loads
    the scaled Stage-A model instead of the full-size one. Both object point sets are the SCALED points, so the
    source interaction mesh, the target/collision points, the collision model and the object pose are ALL at
    scale s -- the scale-consistency fix. Robot<->object non-penetration is ON (RetargeterConfig default).

    Args:
        cfg (PipelineConfig): Pipeline configuration.
        motion_cfg (MotionDataConfig): Motion data configuration.
        con (SimpleNamespace): Task constants.
    Returns:
        str: Path to the Stage-A qpos .npz (key 'qpos', T x nq).
    """
    # PART 0 outputs (both required)
    ref = determine_output_path(cfg, "part0")
    scaled_a = scaled_model_path(cfg, "A")
    require_exists([ref, scaled_a], "run '--stage part0' first")

    z = np.load(ref, allow_pickle=True)
    hj_s, obj_muj, obj_pts_s = z["human_joints_scaled"], z["object_pose_mujoco"], z["object_points_scaled"]
    s = float(z["scale"])

    # point the retargeter at the scaled model (it loads ROBOT_URDF_FILE.replace('.urdf', f'_w_{obj}.xml'))
    con.ROBOT_URDF_FILE = str(scaled_a).replace(f"_w_{cfg.object_name}.xml", ".urdf")
    # tolerance < 0: BOUND the overlap (allow a few mm) rather than force a gap. The G1 elbow/torso capsules
    # touch in the source pose, so a positive clearance is infeasible (self-collision is hard, no slack).
    sc = SelfCollisionConfig(enable=cfg.self_collision, pairs=_ARM_TRUNK_PAIRS, tolerance=-0.005)
    kwargs = build_retargeter_kwargs_from_config(
        retargeter_config=RetargeterConfig(penetration_tolerance=cfg.pen_tolerance, self_collision=sc),
        constants=con,
        object_urdf_path=con.OBJECT_URDF_FILE,
        task_type="object_interaction",
    )
    retargeter = InteractionMeshRetargeter(**kwargs)

    # q_init + MuJoCo-order poses (feed scaled [quat,trans]; initialize_robot_pose converts to [trans,quat])
    obj_qt = obj_muj[:, [3, 4, 5, 6, 0, 1, 2]]
    q_init, _q_nom, obj_aug, hj_s, obj_poses = initialize_robot_pose(
        task_type="object_interaction",
        data_format="smplh",
        human_joints=hj_s,
        object_poses=obj_qt,
        constants=con,
        retargeter=retargeter,
        task_config=TaskConfig(object_name=cfg.object_name),
        augmentation=False,
        save_dir=PKG / cfg.save_dir,
        task_name=cfg.task_name,
    )

    toe = motion_cfg.toe_names
    foot = extract_foot_sticking_sequence_velocity(hj_s, retargeter.demo_joints, toe)
    foot[0][toe[0]] = False
    foot[0][toe[1]] = False

    # both point sets = the scaled points -> source mesh, target, model and pose all at s
    out = determine_output_path(cfg, "part1")

    logger.info("Starting retargeting...")
    retargeter.retarget_motion(
        human_joint_motions=hj_s,
        object_poses=obj_poses,
        object_poses_augmented=obj_aug,
        object_points_local_demo=obj_pts_s,
        object_points_local=obj_pts_s,
        foot_sticking_sequences=foot,
        q_a_init=q_init,
        q_nominal_list=None,
        original=True,
        dest_res_path=out,
    )

    cost = float(np.load(out, allow_pickle=True)["cost"])
    report.part1(cfg, out, len(hj_s), s, str(scaled_a), cost)
    return out


# ============================ PART 2 : floating Wuji-hand-only TopoRetarget ============================

def _body_poses(model, Q, body_name: str) -> np.ndarray:
    """FK every frame; return (T,7) = world pos(3) + quat wxyz(4) of `body_name` (the free hand root)."""
    d = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    out = np.zeros((len(Q), 7))
    for t in range(len(Q)):
        d.qpos[:] = Q[t]
        mujoco.mj_forward(model, d)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.xmat[bid])
        out[t, :3] = d.xpos[bid]
        out[t, 3:] = q
    return out


def _solve_hand(scaled_b, side, O, cfg_kw, hs, obj_quat, obj_pos, Q_init, anchor=0):
    """Worker (runs in its own process): solve ONE hand independently on the S3 manifold base.

    Returns (list of qadr column indices, solved qadr columns) so the parent can merge the two hands
    (their columns are disjoint). MuJoCo isn't thread-safe, so each hand gets its own model + process.
    """
    for p in ("src", "wuji", "geometry_audit"):
        pp = str(PKG / p)
        if pp not in sys.path:
            sys.path.insert(0, pp)
    from holosoma_retargeting.wuji.topoRetarget import TopoRetargeter, human_kp_indices
    from holosoma_retargeting.wuji.hand_table_topo import init_root_from_human

    model = mujoco.MjModel.from_xml_path(scaled_b)
    Q = Q_init.copy()
    ret = TopoRetargeter(model, side, O, **cfg_kw, base_manifold=True)
    init_root_from_human(model, ret, Q, hs, side)                 # Kabsch wrist-root init from the human hand
    Q[:, ret.qadr] = np.clip(Q[:, ret.qadr], ret.lo, ret.hi)
    hkp_mesh = hs[:, human_kp_indices(ret.finger_kp)]
    ret.retarget_fingers(Q, hkp_mesh, obj_quat, obj_pos, anchor=anchor)
    # With base_manifold=True the S3 wrist-ORIENTATION quaternion (the <side>_wrist_ball_joint) is optimized
    # off-qadr and written straight into Q, so ret.qadr covers the wrist TRANSLATION + finger joints but NOT
    # those 4 quaternion columns. They are exactly the gap in qadr (between the wrist-slide cols and the finger
    # cols); return them too, else the caller merges the fingers but leaves the wrist at identity -> the hand
    # flips to point away from the object (base_manifold=False keeps qadr contiguous -> extra is empty, no-op).
    qadr = list(ret.qadr)
    extra = [c for c in range(min(qadr), max(qadr) + 1) if c not in set(qadr)]
    cols = qadr + extra
    return cols, Q[:, cols]


def part2_stageB(cfg: PipelineConfig, motion_cfg: MotionDataConfig, con: SimpleNamespace) -> str:
    """PART 2: floating Wuji-hand-ONLY TopoRetarget -- quaternion (ball) roots, CoACD collision ON (hand+object).

    Each hand is solved INDEPENDENTLY on the SCALED hand-only model (no body), from the canonical scaled hand
    keypoints + object pose. The wrist root is free and Kabsch-initialized from the human hand, so it lands where
    the grasp needs it -- decoupled from Stage-A's body-driven wrist (PART 3 reconciles the two via arm-IK).
    Mirrors geometry_audit/exp_both_hands_stageA_wristroot_s3.py (S3 base). Saves the hand qpos + each hand's
    root pose (T,7) for PART 3.

    Args:
        cfg (PipelineConfig): Pipeline configuration.
        motion_cfg (MotionDataConfig): Motion data configuration.
        con (SimpleNamespace): Task constants.
    Returns:
        str: Path to the Stage-B qpos .npz.
    """
    from concurrent.futures import ProcessPoolExecutor
    for p in ("src", "wuji", "geometry_audit"):                    # research helpers (base_kw + the hand workers)
        pp = str(PKG / p)
        if pp not in sys.path:
            sys.path.insert(0, pp)
    from holosoma_retargeting.geometry_audit.exp_track1_objective import base_kw

    ref = determine_output_path(cfg, "part0")
    require_exists([ref], "run '--stage part0' first")
    z = np.load(ref, allow_pickle=True)
    hk_s = z["hand_kp_scaled"]                                      # (T,2,21,3) scaled world hand keypoints
    obj_muj = z["object_pose_mujoco"]                              # (T,7) [x,y,z, qw,qx,qy,qz] scaled
    O = z["object_points_scaled"]                                 # (P,3) scaled object-local points
    s = float(z["scale"])
    obj_pos, obj_quat = obj_muj[:, :3], obj_muj[:, 3:]

    _, scaled_b, _ = build_scaled_model(cfg, "B", s)              # hand_both_w_<obj>_coacd_ball.xml, scaled
    model = mujoco.MjModel.from_xml_path(scaled_b)
    T = len(hk_s)
    Q = np.zeros((T, model.nq))
    Q[:, model.nq - 7:model.nq - 4] = obj_pos                    # object freejoint (scaled pose, frozen)
    Q[:, model.nq - 4:] = obj_quat

    # winning config (base_kw + both-hands overrides): im-norm lambda500, wrist bone, full-cov tau1mm,
    # base prior 5/1.0, pen_emergency, w_tip10 -- solved on the S3 (quaternion) manifold base.
    cfg_kw = {**base_kw(), **dict(im_normalize=True, lambda_im=500.0, bone_mode="wrist", pen_coverage="full",
              pen_tau=0.001, wrist_reg=0.0, base_reg_trans=5.0, base_reg_rot=1.0, pen_emergency=True, w_tip=10.0,
              obj_name=cfg.object_name)}

    # solve the two hands IN PARALLEL -- independent (disjoint qadr columns), one process each
    logger.info("PART 2 solving both hands in parallel (quaternion root, CoACD collision) ...")
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {side: ex.submit(_solve_hand, scaled_b, side, O, cfg_kw, hk_s[:, si], obj_quat, obj_pos, Q)
                for si, side in enumerate(("left", "right"))}
        for side, fut in futs.items():
            qadr, cols = fut.result()
            Q[:, qadr] = cols                                     # merge this hand's solved columns

    root_l = _body_poses(model, Q, "hand_root_left")             # per-hand root pose (T,7) for PART 3 arm-IK
    root_r = _body_poses(model, Q, "hand_root_right")

    out = determine_output_path(cfg, "part2")
    np.savez(out, qpos=Q, object_points=O, hand_root_left=root_l, hand_root_right=root_r, fps=30,
             human_joints=z["human_joints_scaled"])
    report.part2(cfg, out, T, s, scaled_b)
    return out


# ============================ PART 3 : elbow-aware arm + contact-aware composition ============================

def _jointmap(model) -> dict:
    """{joint name: (qposadr, nqpos)} for every joint (7 free, 4 ball, else 1)."""
    out = {}
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        n = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(model.jnt_type[j], 1)
        out[nm] = (int(model.jnt_qposadr[j]), n)
    return out


_ARM_J = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
_CFG_ELBOW = dict(im_normalize=True, lambda_im=500.0, bone_mode="wrist", pen_coverage="full", pen_tau=0.001,
                  base_reg_trans=0.0, base_reg_rot=0.0, pen_emergency=True, w_tip=10.0,
                  use_smooth=True, w_smooth=0.5, vel_bound=True, vmax_rot=0.12, vmax_trans=0.05)


def _solve_arm(full_xml, side, O, hs, obj_quat, obj_pos, Q_init, sched, anchor, obj_name="table"):
    """Worker (own process): elbow-aware TopoRetarget for ONE arm+hand -- base_joints = the 7 G1 arm joints,
    base_manifold=False, body frozen. Phase-aware schedules make the arm track the grasp in contact and return
    toward the Stage-A arm off-contact (fingers open off-contact). Returns (qadr indices, solved columns).
    """
    for p in ("src", "wuji", "geometry_audit"):
        pp = str(PKG / p)
        if pp not in sys.path:
            sys.path.insert(0, pp)
    from holosoma_retargeting.wuji.topoRetarget import TopoRetargeter, human_kp_indices
    from holosoma_retargeting.geometry_audit.exp_track1_objective import base_kw

    m = mujoco.MjModel.from_xml_path(full_xml)
    Q = Q_init.copy()
    armj = [f"{side}_{j}_joint" for j in _ARM_J]
    ret = TopoRetargeter(m, side, O, **{**base_kw(), **_CFG_ELBOW, "wrist_reg": 0.0, "base_bound": None, "obj_name": obj_name},
                         base_joints=armj, base_manifold=False)
    ret.wrist_reg_sched, ret.lambda_im_sched, ret.finger_reg_sched = sched
    ret.q_finger_ref = m.qpos0[[ret.qadr[i] for i in range(ret.n_wrist, len(ret.qadr))]]   # neutral/open fingers
    ret.retarget_fingers(Q, hs[:, human_kp_indices(ret.finger_kp)], obj_quat, obj_pos, anchor=anchor)
    return list(ret.qadr), Q[:, ret.qadr]


def part3_assemble(cfg: PipelineConfig, motion_cfg: MotionDataConfig, con: SimpleNamespace) -> str:
    """PART 3: elbow-aware arm + contact-aware composition (emits TWO finals to compare).

    Runs ONE PURE integrated TopoRetarget per hand (real 7-DOF G1 arm + fingers, body frozen from Stage-A,
    CONSTANT weights -- no in-solve schedules), then a POST-HOC contact/release blend (arm: Topo in contact ->
    Omni resting off-contact over 8 frames). Composes:
      COMBINE  = Stage-A body/object + blended arm + (Topo grasp -> open) fingers.
      SEPARATE = Stage-A body/object + blended arm + Stage-B (S3) fingers.

    Args:
        cfg (PipelineConfig): Pipeline configuration.
        motion_cfg (MotionDataConfig): Motion data configuration.
        con (SimpleNamespace): Task constants.
    Returns:
        str: Path to the COMBINE final .npz (SEPARATE saved alongside).
    """
    from concurrent.futures import ProcessPoolExecutor
    for p in ("src", "wuji", "geometry_audit"):
        pp = str(PKG / p)
        if pp not in sys.path:
            sys.path.insert(0, pp)
    from holosoma_retargeting.geometry_audit.wrist_bone_diag import contacts

    ref = determine_output_path(cfg, "part0")
    stageA = determine_output_path(cfg, "part1")
    stageB = determine_output_path(cfg, "part2")
    require_exists([ref, stageA, stageB], "run part0, part1 (Stage-A) and part2 (Stage-B) first")
    z = np.load(ref, allow_pickle=True)
    hk_s, obj_muj, O, s = z["hand_kp_scaled"], z["object_pose_mujoco"], z["object_points_scaled"], float(z["scale"])
    obj_pos, obj_quat = obj_muj[:, :3], obj_muj[:, 3:]

    full_base = PKG / "models" / cfg.robot / f"{cfg.robot}_29dof_wuji_w_{cfg.object_name}_coacd.xml"
    if not full_base.exists():
        raise ValueError(f"full Wuji model not found: {full_base}")
    full_scaled = str(PKG / cfg.save_dir / "models" / cfg.task_name / "final" / full_base.name)
    build_scaled_object_model(str(full_base), s, full_scaled, cfg.object_name)

    mf = mujoco.MjModel.from_xml_path(full_scaled)
    mw = mujoco.MjModel.from_xml_path(str(scaled_model_path(cfg, "A")))
    mh = mujoco.MjModel.from_xml_path(str(scaled_model_path(cfg, "B")))
    qw = np.load(stageA, allow_pickle=True)["qpos"]
    qh = np.load(stageB, allow_pickle=True)["qpos"]
    T = min(len(qw), len(qh))
    Jf, Jw, Jh = _jointmap(mf), _jointmap(mw), _jointmap(mh)

    # Q init: Stage-A body/arm/object (by name) + S3 fingers (by name). The arm columns are the Omni arm reference.
    Q = np.tile(mf.qpos0, (T, 1))
    for nm, (a, n) in Jf.items():
        if nm is None:
            continue
        if "finger" in nm and nm in Jh:
            Q[:, a:a + n] = qh[:T, Jh[nm][0]:Jh[nm][0] + n]
        elif nm in Jw:
            Q[:, a:a + n] = qw[:T, Jw[nm][0]:Jw[nm][0] + n]
    Q[:, 0:7] = qw[:T, 0:7]
    Q[:, mf.nq - 7:mf.nq] = qw[:T, mw.nq - 7:mw.nq]

    # contact mask + 8-frame release ramp from the human hand-object distance (both hands).
    # Use the ACTUAL object mesh scaled by s (NOT a hardcoded table) so contact distances match the scaled frame.
    om_mesh = str(PKG / "models" / cfg.object_name / f"{cfg.object_name}.obj")
    cmL, _ = contacts(hk_s[:, 0], obj_pos, obj_quat, T, obj_mesh=om_mesh, scale=s)
    cmR, _ = contacts(hk_s[:, 1], obj_pos, obj_quat, T, obj_mesh=om_mesh, scale=s)
    contact = (cmL.sum(1) > 0) | (cmR.sum(1) > 0)
    dist = np.where(contact, 0.0, 1e9)
    for i in range(1, T):
        dist[i] = min(dist[i], dist[i - 1] + 1)
    for i in range(T - 2, -1, -1):
        dist[i] = min(dist[i], dist[i + 1] + 1)
    snc = np.clip(dist / 8.0, 0.0, 1.0)                           # 0 in contact -> 1 deep non-contact (8-frame ramp)
    s_blend = 1.0 - snc                                          # post-hoc blend weight: 1 in contact -> 0 off-contact
    # anchor (Kabsch init frame) = MIDDLE of the longest contiguous contact run, computed PER HAND from that
    # hand's own contact (the solve is per-hand and each arm's stable-branch window differs). The pure solve is
    # highly anchor-sensitive: a mid-contact frame (stable, unambiguous grasp) seeds the correct elbow branch
    # (~60 flips), while contact onset/offset frames seed a bad branch (~500 flips). See diag_anchor_sweep.py.
    def _anchor(cm_hand):
        c = cm_hand.sum(1) > 0
        if not c.any():
            return T // 2
        idx = np.where(c)[0]
        run = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
        return int(run[len(run) // 2])
    anchor = {"left": _anchor(cmL), "right": _anchor(cmR)}
    logger.info("PART 3 pure integrated solve + post-hoc contact/release blend | contact %d/%d frames | anchor L%d R%d",
                int(contact.sum()), T, anchor["left"], anchor["right"])

    # PURE integrated solve: CONSTANT weights, NO in-solve schedules (the schedules caused the left-arm branch
    # flips: 403->55 flips + 7.1->1.5cm tracking when disabled). Contact/release handled by the post-hoc blend below.
    sched_pure = (np.zeros(T), np.full(T, 500.0), np.zeros(T))   # wrist_reg=0, lambda_im=500, finger_reg=0
    Q_omni = Q.copy()                                            # OmniRetarget resting arm + S3 fingers (blend targets)
    open_ref = mf.qpos0                                          # open-hand reference (neutral fingers)
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {side: ex.submit(_solve_arm, full_scaled, side, O, hk_s[:, si], obj_quat, obj_pos, Q, sched_pure, anchor[side], cfg.object_name)
                for si, side in enumerate(("left", "right"))}
        solved = {side: fut.result() for side, fut in futs.items()}

    # POST-HOC contact/release blend (interpolates already-solved angles -- NOT another solve):
    #   arm             q = s*Topo + (1-s)*Omni          (identical for both variants)
    #   COMBINE fingers q = s*Topo_grasp + (1-s)*open    (fingers open off-contact)
    #   SEPARATE fingers = demonstrated S3 trajectory as-is (opens on its own with the human hand)
    combine, separate = Q_omni.copy(), Q_omni.copy()
    for side in ("left", "right"):
        qadr, cols = solved[side]
        arm_adr = {Jf[f"{side}_{j}_joint"][0] for j in _ARM_J}
        for ci, a in enumerate(qadr):
            if a in arm_adr:                                     # arm: blend Topo <-> Omni (both variants)
                blend = s_blend * cols[:, ci] + (1.0 - s_blend) * Q_omni[:, a]
                combine[:, a] = blend
                separate[:, a] = blend
            else:                                                # COMBINE finger: blend Topo grasp <-> open ref
                combine[:, a] = s_blend * cols[:, ci] + (1.0 - s_blend) * open_ref[a]
    # SEPARATE fingers already = S3 (from the name-mapped init), left untouched

    hj = z["human_joints_scaled"][:T]
    out = determine_output_path(cfg, "part3")
    out_c, out_s = out.replace("_final.npz", "_final_combine.npz"), out.replace("_final.npz", "_final_separate.npz")
    np.savez(out_c, qpos=combine, fps=30, human_joints=hj)
    np.savez(out_s, qpos=separate, fps=30, human_joints=hj)
    report.part3(cfg, out_c, out_s, T, int(contact.sum()), full_scaled)
    return out_c


def main(cfg: PipelineConfig) -> None:
    validate_config(cfg)
    if cfg.pipeline_version == "v2" and cfg.stage in ("part3", "part4"):
        raise ValueError("V2 directly exports its refined motion in part2; use --stage part2 or all")
    logger.info("Pipeline %s | task: %s | stage: %s | data_path: %s", cfg.dataset, cfg.task_name, cfg.stage, cfg.data_path)

    # Stage-A end-effector: wuji -> smplh_wuji_body (wrist->Wuji palm); rubber -> smplh (wrist->rubber_hand).
    data_format = "smplh_wuji_body" if cfg.stageA_hand == "wuji" else "smplh"
    motion_cfg = MotionDataConfig(data_format=data_format, robot_type=cfg.robot)
    con = create_task_constants(
        robot_config=RobotConfig(robot_type=cfg.robot),
        motion_data_config=motion_cfg,
        task_config=TaskConfig(object_name=cfg.object_name),
        task_type="object_interaction",
    )

    if cfg.stage in ("part0", "all"):
        part0(cfg, motion_cfg, con)                                 # canonical reference (.npz) + scaled Stage-A model (.xml)
    if cfg.stage in ("part1", "all"):
        part1_stageA(cfg, motion_cfg, con)                          # OmniRetarget Stage-A whole-body qpos (.npz)
    if cfg.pipeline_version == "v2":
        if cfg.stage in ("part2", "all"):
            import subprocess
            ref = determine_output_path(cfg, "part0")
            body = determine_output_path(cfg, "part1")
            require_exists([ref, body], "V2 requires part0 and OmniRetarget part1")
            scale = float(np.load(ref, allow_pickle=False)["scale"])
            full_base = PKG / "models" / cfg.robot / f"{cfg.robot}_29dof_wuji_w_{cfg.object_name}_coacd.xml"
            full_scaled = PKG / cfg.save_dir / "models" / cfg.task_name / "final" / full_base.name
            build_scaled_object_model(str(full_base), scale, str(full_scaled), cfg.object_name)
            destination = PKG / cfg.save_dir / "v2"
            subprocess.run([sys.executable, str(Path(__file__).with_name("pipeline_v2.py")),
                "--canonical", str(ref), "--body", str(body), "--body-model", str(scaled_model_path(cfg, "A")),
                "--model", str(full_scaled), "--object", cfg.object_name, "--output", str(destination)], check=True)
            converter = PKG.parents[1] / "wuji_wbt" / "motion" / "convert_wuji.py"
            subprocess.run([sys.executable, str(converter), str(destination / "final_v2.npz"),
                str(full_scaled), str(destination / "v2_wuji_50hz.npz"), "30", "50"], check=True)
        return
    if cfg.stage in ("part2", "all"):
        part2_stageB(cfg, motion_cfg, con)                          # floating Wuji-hand-only TopoRetarget (.npz)
    if cfg.stage in ("part3", "all"):
        part3_assemble(cfg, motion_cfg, con)                        # name-mapped whole-body assembly (.npz)
    if cfg.stage == "part4":
        logger.info("Stage %s not yet wired in this build step.", cfg.stage)


if __name__ == "__main__":
    cfg = tyro.cli(PipelineConfig)
    main(cfg)
