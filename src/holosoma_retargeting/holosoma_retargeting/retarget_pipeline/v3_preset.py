"""Tuned V3: shared weights validated in the 14-window HUMOTO/HiPHI pilot.

Use with the collision-aware initializer, geometry-side contact, regional
surface targets, signed recovery disabled, and MuJoCo collision backend.
This preset changes weights, not collision tolerances or evaluation thresholds.
It is provisional: the pilot does not establish physical grasp success.
"""

VERSION = "v3"
NAME = "Tuned V3"
EXPERIMENT_VARIANT = "surface_emphasis"
SURFACE_GAP_WEIGHT = 1024.0
FINGERTIP_WEIGHT = 25.0


def apply_weights(refiner, extra):
    """Apply the exact tested weights to the coupled solver and its extra terms."""
    refiner.cfg.fingertip_weight = FINGERTIP_WEIGHT
    extra["gap_weight"] = SURFACE_GAP_WEIGHT
