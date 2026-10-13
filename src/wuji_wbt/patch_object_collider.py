"""Import interaction-object URDFs with convex decomposition.

The objects_ filename prefix identifies interaction objects. Their USDs are
regenerated so changed collision settings are not hidden by a cached asset.
Apply after Isaac Lab startup and before constructing the environment.
"""

from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED  # noqa: PLW0603 - apply each runtime patch once per process.
    if _APPLIED:
        return
    import isaaclab.sim as sim_utils

    Cfg = sim_utils.UrdfFileCfg
    _orig_post = getattr(Cfg, "__post_init__", None)

    def _post_init(self):
        ap = str(getattr(self, "asset_path", "") or "")
        if "objects_" in ap:  # the interaction object, not the robot
            object.__setattr__(self, "collider_type", "convex_decomposition")
            object.__setattr__(self, "force_usd_conversion", True)
        if _orig_post is not None:
            _orig_post(self)

    Cfg.__post_init__ = _post_init
    _APPLIED = True
    print("[patch_object_collider] object URDFs -> convex_decomposition + force_usd_conversion")
