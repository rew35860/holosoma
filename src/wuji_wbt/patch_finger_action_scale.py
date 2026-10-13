"""Use 0.25 rad per policy action for the original Wuji finger joints.

The stock effort/kp rule remains active for the body. Apply this patch before
the action manager is built; train_wuji.py does so after simulator startup.
"""

from __future__ import annotations

FINGER_ACTION_SCALE = 0.25  # rad per unit action (body DOFs get 0.35-0.55 from the stock rule)
_APPLIED = False


def apply(scale: float = FINGER_ACTION_SCALE) -> None:
    global _APPLIED  # noqa: PLW0603 - apply each runtime patch once per process.
    if _APPLIED:
        return
    from holosoma.managers.action.terms import joint_control as JC

    _orig = JC.JointPositionActionTerm._configure_action_scales

    def _configure(self, env):
        _orig(self, env)
        for i, n in enumerate(env.robot_config.dof_names):
            if "finger" in n:
                self.action_scales[i] = scale

    JC.JointPositionActionTerm._configure_action_scales = _configure
    _APPLIED = True
    print(f"[patch_finger_action_scale] finger DOF action scale -> {scale} rad/unit")
