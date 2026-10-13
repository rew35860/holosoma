"""Domain randomization for a single-clip carry policy = holosoma's body+object WBT randomization with
three object/push knobs re-sized for a hand-held object:
  object mass      URDF nominal +-0.3 kg   (stock ADDS 1-4 kg, sized for the 0.1 kg table)
  object friction  static 0.6-1.0, dynamic 0.5-0.9, restitution 0-0.2   (stock 0.1-0.6 makes a hook grip slip)
  pushes           0.3 m/s every 2-4 s     (stock 0.5 m/s every 1-3 s knocks a two-hand carry loose)
Everything else (robot material/COM/dof bias, actuator DR, action delay) is the stock config.
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization_w_object as _stock


def _with_params(term: RandomizationTermCfg, **params) -> RandomizationTermCfg:
    return replace(term, params={**term.params, **params})


_setup = dict(_stock.setup_terms)
_setup["randomize_object_rigid_body_mass_startup"] = _with_params(
    _setup["randomize_object_rigid_body_mass_startup"], mass_distribution_params=[-0.3, 0.3]
)
_setup["randomize_object_rigid_body_material_startup"] = _with_params(
    _setup["randomize_object_rigid_body_material_startup"],
    static_friction_range=[0.6, 1.0],
    dynamic_friction_range=[0.5, 0.9],
    restitution_range=[0.0, 0.2],
)
_setup["push_randomizer_state"] = _with_params(
    _setup["push_randomizer_state"], push_interval_s=[2.0, 4.0], max_push_vel=[0.3, 0.3, 0.1, 0.3, 0.3, 0.5]
)

wuji_wbt_randomization_w_object = replace(_stock, setup_terms=_setup)
