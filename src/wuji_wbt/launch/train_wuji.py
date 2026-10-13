"""Train one HUMOTO body/hand/object policy with the original Wuji robot.

Preset registration lives in configs/experiment_wuji.py; reward weights and
sigmas live in configs/reward_wuji.py. This launcher applies object convex
decomposition and the finger action scale before Holosoma builds the environment.

From the repository root in the hssim environment:
    python src/wuji_wbt/launch/train_wuji.py exp:wuji-humoto-648-v7 logger:disabled \
        --training.headless=True \
        --reward.terms.fingertip-pos-object-error-exp.weight=6.0

Use --help after the preset for available reward and training overrides.
"""

import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../src on path -> import wuji_wbt

import tyro
import tyro.conf
import tyro.extras

import holosoma.train_agent as TA
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.train_agent import train
from holosoma.utils.tyro_utils import TYRO_CONIFG
from wuji_wbt.configs.experiment_wuji import PRESETS
from wuji_wbt.patch_finger_action_scale import apply as _apply_finger_action_scale
from wuji_wbt.patch_object_collider import apply as _apply_object_decomp

_orig_init = TA.init_sim_imports


def _init_then_patch(*a, **k):
    app = _orig_init(*a, **k)  # SimulationApp up -> isaaclab.sim importable
    _apply_object_decomp()  # object URDFs -> convex_decomposition (concave collision)
    _apply_finger_action_scale()  # finger DOFs -> usable action scale
    return app


TA.init_sim_imports = _init_then_patch

AnnotatedCfg = Annotated[
    ExperimentConfig,
    tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(PRESETS)),
]


def main() -> None:
    cfg = tyro.cli(AnnotatedCfg, config=TYRO_CONIFG)
    train(cfg)


if __name__ == "__main__":
    main()
