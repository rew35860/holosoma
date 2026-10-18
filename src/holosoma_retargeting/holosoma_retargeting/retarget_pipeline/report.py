"""Console reports for the retargeting pipeline (kept out of pipeline.py for readability).

Pure formatting/logging -- no computation, no asserts. Each `part<N>` logs the stage summary the
main flow hands it, through the shared logger (so verbosity is controlled the same way as everywhere else).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def part0(cfg, task_name, T, s, z_min, human_height, stand, off, ref) -> None:
    """PART 0 canonical-reference report. `off` = dict from checks.assert_contact_offsets_preserved."""
    line = "=" * 84
    logger.info(line)
    logger.info("PART 0  canonical scaled reference  |  %s:%s  |  T=%d", cfg.dataset, task_name, T)
    logger.info(line)
    logger.info("  scale s = %.6f   z_min(floor) = %.4f m   (transform: s*(raw-[0,0,z_min]))", s, z_min)
    logger.info("  implied raw human_height = %.3f m (bent-pose z-span %.2f m) -> single-scale [asserted]",
                human_height, stand)
    logger.info("  object_local scaled == s*raw  [asserted]   object_pose MuJoCo-order (same src/tgt/coll/render)")
    if off is None:
        logger.info("  no hand keypoints (body-only run) -- contact-offset check skipped")
    else:
        logger.info("  CONTACT-OFFSET PRESERVATION (uniform scaling scales offsets by s):")
        logger.info("    right wrist->object : raw %+.4f -> scaled %+.4f m (expect %+.4f) err %.1e [asserted]",
                    off["w_raw"].mean(), off["w_s"].mean(), s * off["w_raw"].mean(), off["dw"])
        logger.info("    right fingertip     : raw %+.4f -> scaled %+.4f m (expect %+.4f) err %.1e [asserted]",
                    off["f_raw"].mean(), off["f_s"].mean(), s * off["f_raw"].mean(), off["df"])
    logger.info("  -> reference (data + provenance in one file): %s", ref)  # 11 spaces = len('INFO: ')+'  -> '
    logger.info("  DONE_PART0\n")


def part0_model(cfg, scaled_path, n_scaled, s, geo) -> None:
    """PART 0 scaled-object-model report. `geo` = dict from checks.assert_object_model_scaled."""
    obj = cfg.object_name
    line = "=" * 84
    logger.info(line)
    logger.info("PART 0  scaled Stage-A %s model  |  %d %s meshes x s=%.4f  (%d CoACD pieces)",
                obj, n_scaled, obj, s, geo["n_col"])
    logger.info(line)
    logger.info("  %s origin (default pose): [%+.3f %+.3f %+.3f] m", obj, *geo["origin"])
    logger.info("  %s-top height above origin: base %.4f -> scaled %.4f m (expect %.4f) [asserted]",
                obj, geo["base_vis_top"] - geo["origin"][2], geo["vis_top"] - geo["origin"][2],
                s * (geo["base_vis_top"] - geo["origin"][2]))
    logger.info("  scaled visible top %.4f m  |  CoACD collision top %.4f m", geo["vis_top"], geo["col_top"])
    logger.info("  visible-vs-collision gap: base %.2f -> scaled %.2f mm (scales x s, no new under-fill) [asserted]",
                geo["vc_base"] * 1e3, geo["vc_scaled"] * 1e3)
    logger.info("  -> scaled model (PART 1 collides against this): %s", scaled_path)
    logger.info("  DONE_PART0_MODEL\n")


def part1(cfg, out, T, s, model, cost) -> None:
    """PART 1 OmniRetarget Stage-A report."""
    line = "=" * 84
    logger.info(line)
    logger.info("PART 1  OmniRetarget Stage-A whole body (body + wrist)  |  %s  |  T=%d", cfg.task_name, T)
    logger.info(line)
    logger.info("  scale s=%.4f -> source mesh, target points, collision model AND object pose ALL at s", s)
    logger.info("  scaled model (robot<->object non-penetration ON): %s", model)
    logger.info("  final-frame solver cost: %.4f", cost)
    logger.info("  -> Stage-A qpos (key 'qpos', T x nq): %s", out)
    logger.info("  DONE_PART1\n")


def part2(cfg, out, T, s, model) -> None:
    """PART 2 floating Wuji-hand-only TopoRetarget report."""
    line = "=" * 84
    logger.info(line)
    logger.info("PART 2  floating Wuji-hand-only TopoRetarget (quaternion roots)  |  %s  |  T=%d", cfg.task_name, T)
    logger.info(line)
    logger.info("  each hand solved independently at s=%.4f; free wrist root (Kabsch init), decoupled from Stage-A", s)
    logger.info("  config: im-norm lambda500 + wrist-bone + full-cov tau1mm + base 5/1.0 + pen_emergency + w_tip10")
    logger.info("  scaled hand-only model (ball roots, CoACD hand+object collision ON): %s", model)
    logger.info("  -> Stage-B qpos + per-hand root pose (keys 'qpos','hand_root_left','hand_root_right'): %s", out)
    logger.info("  DONE_PART2\n")


def part3(cfg, out_combine, out_separate, T, n_contact, model) -> None:
    """PART 3 pure integrated solve + post-hoc contact/release blend report (two finals to compare)."""
    line = "=" * 84
    logger.info(line)
    logger.info("PART 3  pure integrated solve + post-hoc blend  |  %s  |  T=%d  (contact %d frames)",
                cfg.task_name, T, n_contact)
    logger.info(line)
    logger.info("  body/legs/feet/object <- Stage-A; arm <- PURE TopoRetarget (constant weights, no in-solve schedules)")
    logger.info("  post-hoc blend (8-frame ramp): arm s*Topo+(1-s)*Omni; COMBINE fingers s*grasp+(1-s)*open")
    logger.info("  final model (full body + Wuji fingers, scaled): %s", model)
    logger.info("  -> COMBINE  (blended arm + grasp->open fingers): %s", out_combine)
    logger.info("  -> SEPARATE (blended arm + S3 fingers)         : %s", out_separate)
    logger.info("  DONE_PART3\n")
