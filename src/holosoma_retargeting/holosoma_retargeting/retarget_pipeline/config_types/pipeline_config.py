"""Top-level config for the scale-consistent whole-body G1+Wuji retargeting pipeline (Tyro CLI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class PipelineConfig:
    """Configuration for one whole-body retargeting run (used by the Tyro CLI)."""

    task_name: str = "push_table_forward_with_both_hands-039"
    """Sequence / task name (the <task_name>.pt + <task_name>_handkp.npy stem)."""

    pipeline_version: Literal["v1", "v2"] = "v1"
    """v1 preserves the archived floating-hand/SEPARATE pipeline. v2 uses quick
    finger initialization, one coupled 54-DOF arm/finger solve, and direct export.
    For v2 use stage=all (including body) or part2 (reuse completed part0/part1)."""

    dataset: str = "humoto"
    """Source dataset. 'humoto' now; other InterAct-format datasets later. Selects the raw data layout."""

    robot: str = "g1"
    """Robot type (drives robot_height / scale via _ROBOT_DEFAULTS)."""

    stageA_hand: Literal["wuji", "rubber"] = "wuji"
    """Stage-A (PART 1) end-effector. 'wuji' -> Wuji-welded model + smplh_wuji_body mapping (wrist->Wuji palm);
    'rubber' -> rubber-mitten model + smplh mapping (wrist->rubber_hand). Only affects PART 1's arm placement;
    PART 2/3 are independent of it."""

    stage: Literal["part0", "part1", "part2", "part3", "part4", "all"] = "part0"
    """Which pipeline stage(s) to run. part0 = canonical scaled reference + scaled table model;
    part1 = OmniRetarget Stage-A body; part2 = attached TopoRetarget hands; part3 = assembly;
    part4 = validation + render; all = the whole thing end-to-end."""

    # --- inputs ---
    data_path: Path | None = None
    """Directory holding the raw <task_name>.pt (human + object) and <task_name>_handkp.npy (hand keypoints).
    Lives OUTSIDE the project -> defaults to None; pass it at runtime with --data-path <path>."""

    object_name: str = "table"
    """Interaction object name (drives task constants -> the canonical object mesh/urdf, no separate path)."""

    # --- outputs ---
    save_dir: Path = Path("demo_results/g1/retarget_pipeline")
    """Directory for all pipeline artifacts (project-relative)."""

    n_obj: int = 100
    """Object surface sample count for the interaction mesh."""

    pen_tolerance: float = 0.001
    """PART 1 robot<->object non-penetration tolerance (m). 1mm default; raise it (e.g. 0.02) for sequences
    where OmniRetarget's hard non-penetration constraint goes infeasible (a body-object collision it can't resolve)."""

    self_collision: bool = False
    """PART 1 (Stage-A) arm<->trunk self-collision avoidance. Off by default (OmniRetarget's default). When True,
    enforces a hard min-distance (1cm) between the forearm links (elbow/wrist) and the torso/waist/pelvis so the
    arm cannot pass through the body. Only affects PART 1; PART 3 re-solves the arm and would need its own term."""

    # --- models (base, unscaled; PART 0 generates the scaled variants used by every stage) ---
    stageA_model_base: Path | None = None
    """Stage-A body model (rubber-mitten wrist + object CoACD collision). None -> auto-resolve
    models/<robot>/<robot>_29dof_w_<object_name>.xml. Pass a project-relative path to override."""

    stageB_model_base: Path | None = None
    """Stage-B model (robot + Wuji fingers + object). None -> auto-resolve
    models/<robot>/<robot>_29dof_wuji_w_<object_name>.xml. Pass a project-relative path to override."""
