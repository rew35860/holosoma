"""HUMOTO clip registry for one-policy-per-sequence training.

Paths derive from this repository. Each entry defines the task, object,
nominal mass, optional reference suffix and friction override. Generated
motion files and object assets remain local data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # .../holosoma
RETARGET = REPO / "src" / "holosoma_retargeting" / "holosoma_retargeting"
WBT_DATA = "holosoma/data/motions/g1_29dof/whole_body_tracking"  # holosoma package-data prefix
WBT_DATA_DIR = REPO / "src" / "holosoma" / WBT_DATA


@dataclass(frozen=True)
class SequenceSpec:
    name: str  # preset suffix: exp:wuji-<name>
    task: str  # retarget task name (part0/part3 file stem)
    obj: str  # object name: models/<obj>/<obj>.obj, objects_<obj>.urdf
    object_mass_kg: float  # nominal mass written into the object URDF (DR adds +-0.3 kg)
    motion_tag: str = ""  # "" = the kinematic retarget; "_spider" = the SPIDER-refined reference (spider_bridge/)
    urdf_tag: str = ""  # "" = objects_<obj>.urdf; a tag gives this sequence its own URDF (e.g. a lighter mass)
    object_friction: tuple[float, float] | None = None  # static-friction DR range override (default 0.6-1.0)

    @property
    def object_urdf(self) -> str:
        """IsaacSim object URDF (built by tools/make_object_urdf.py)."""
        return f"{WBT_DATA}/objects_{self.obj}{self.urdf_tag}.urdf"

    @property
    def retarget_npz(self) -> Path:
        """PART 3 output: 83-dof qpos (root7 + 69 joints + object7) on the scaled final model, 30 fps."""
        return RETARGET / "demo_results/g1/retarget_pipeline/part3" / f"{self.task}_final_separate.npz"

    @property
    def canonical_npz(self) -> Path:
        """PART 0 output: holds the HUMOTO->G1 uniform scale used for the object mesh."""
        return RETARGET / "demo_results/g1/retarget_pipeline/part0" / f"{self.task}_canonical.npz"

    @property
    def mjcf(self) -> Path:
        """Scaled G1+Wuji+object MJCF: FK for the conversion and MuJoCo rendering of rollouts."""
        return RETARGET / "models/scaled/final" / f"g1_29dof_wuji_w_{self.obj}_coacd.xml"

    @property
    def obj_mesh(self) -> Path:
        return RETARGET / "models" / self.obj / f"{self.obj}.obj"

    @property
    def motion_npz(self) -> Path:
        """50 Hz holosoma motion with the 69 joints + object (motion/convert_wuji.py output)."""
        return (
            RETARGET / "demo_results/g1/retarget_pipeline/wbt_motions" / f"{self.task}_wuji_50hz{self.motion_tag}.npz"
        )


# Nominal masses are simulation settings; generated object URDFs must use the same values.
SEQUENCES: dict[str, SequenceSpec] = {
    "humoto-737": SequenceSpec("humoto-737", "carrying_dining_chair_with_both_hands-737", "dining_chair", 2.5),
    "humoto-039": SequenceSpec("humoto-039", "push_table_forward_with_both_hands-039", "table", 3.0),
    "humoto-039-spider": SequenceSpec(
        "humoto-039-spider", "push_table_forward_with_both_hands-039", "table", 3.0, "_spider"
    ),
    "humoto-708": SequenceSpec("humoto-708", "lifting_and_moving_table_with_both_hands-708", "table", 3.0),
    "humoto-399": SequenceSpec("humoto-399", "lift_table_with_both_hands_and_put_on_ground-399", "table", 3.0),
    "humoto-105": SequenceSpec("humoto-105", "lifting_and_carrying_table-105", "table", 3.0),
    "humoto-648": SequenceSpec(
        "humoto-648", "lifting_dining_chair_and_placing_back_with_both_hands_on_ground-648", "dining_chair", 2.5
    ),
    "humoto-266": SequenceSpec("humoto-266", "walk_with_dining_chair-266", "dining_chair", 2.5),
    "humoto-039-spider-light": SequenceSpec(
        "humoto-039-spider-light",
        "push_table_forward_with_both_hands-039",
        "table",
        1.5,
        "_spider",
        "_1p5kg",
        (0.4, 0.7),
    ),
}
