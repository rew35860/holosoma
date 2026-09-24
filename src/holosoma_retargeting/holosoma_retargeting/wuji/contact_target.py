#!/usr/bin/env python
"""Extract hand-object CONTACT TARGETS from a HUMOTO demo — where the human touches the object.

This is the piece neither OmniRetarget nor TopoRetarget has: they scatter object points EVENLY over
the whole mesh and hope the interaction mesh couples the hand to them, but nothing tells the retarget
WHERE the human actually made contact. The finger stage cannot fix a hand that Stage A parked ~1.5 cm
off the rim, because it frees only wrist ROTATION + finger curl -- never the hand's POSITION. So to put
the robot hand on the rim we first have to know where the rim contact IS. That is what this produces.

Per frame, per hand, for each fingertip:
  * transform the human keypoint into the object-local frame (the SAME convention the retargeter uses,
    via transform_points_world_to_local), so a contact defined here lines up with the solve;
  * find the nearest point on the object SURFACE -- this, not the raw human fingertip, is the target:
    the human fit can sit slightly inside or outside the mesh, and we want "land ON the surface here",
    not "reproduce the human's penetration/hover";
  * flag it as contact when that distance is under a threshold.

Distances are UNSIGNED (nearest-surface-point), because HUMOTO meshes are not watertight (0/72) and a
signed query is unreliable on them -- the same trap that made mesh.contains()/signed_distance wrong
before. "In contact" therefore means "within `tau` of the surface", not "inside".

Writes an .npz the wrist-placement stage consumes:
    tgt_world (T,2,5,3)  nearest surface point per fingertip, WORLD frame  (the placement target)
    tgt_local(T,2,5,3)  same, object-local frame
    contact  (T,2,5)     bool: fingertip within `tau` of the surface
    dist     (T,2,5)     fingertip -> surface distance (m)
    tips                 the MediaPipe fingertip indices used, for provenance

Usage (omniretarget env), run from the retargeting root:
    python wuji/contact_target.py \
        demo_results/.../<seq>_original.npz models/table/table.obj \
        ../../../../InterAct/result/humoto_pt/<seq>_handkp.npy [--tau 0.03] [--out <seq>_contact.npz]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from utils import transform_points_world_to_local  # noqa: E402

TIPS = [4, 8, 12, 16, 20]        # MediaPipe fingertips: thumb, index, middle, ring, pinky
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HANDS = ["left", "right"]


def object_pose(stageA_npz: str):
    """Per-frame object pose from the Stage-A trajectory (the object freejoint at the tail of qpos).

    The object is the last freejoint, so its 7 values sit at the end of every qpos row: [xyz, wxyz].
    Reading it here rather than re-deriving keeps this identical to what the retargeter saw.

    Args:
        stageA_npz: the Stage-A `*_original.npz` (welded body solve).

    Returns:
        tuple: (pos (T,3), quat (T,4) wxyz).
    """
    q = np.load(stageA_npz, allow_pickle=True)["qpos"]
    return q[:, -7:-4].copy(), q[:, -4:].copy()


def fingertip_contacts(handkp: np.ndarray, mesh: trimesh.Trimesh,
                       obj_pos: np.ndarray, obj_quat: np.ndarray, tau: float):
    """Nearest surface point + contact flag for every fingertip, every hand, every frame.

    Args:
        handkp: (T, 2, 21, 3) human hand keypoints, WORLD frame, [left, right], MediaPipe order.
        mesh: the object mesh in its LOCAL frame.
        obj_pos: (T, 3) object translation.
        obj_quat: (T, 4) object orientation (wxyz).
        tau: contact threshold (m) -- a fingertip within this of the surface counts as contact.

    Returns:
        tuple: (tgt_world (T,2,5,3), tgt_local (T,2,5,3), contact (T,2,5) bool, dist (T,2,5)).
    """
    pq = trimesh.proximity.ProximityQuery(mesh)
    T = len(handkp)
    tgt_local = np.zeros((T, 2, 5, 3), np.float32)
    tgt_world = np.zeros((T, 2, 5, 3), np.float32)
    dist = np.zeros((T, 2, 5), np.float32)

    for t in range(T):
        for h in range(2):
            tips_world = handkp[t, h, TIPS]                                  # (5,3) world
            tips_local = transform_points_world_to_local(obj_quat[t], obj_pos[t], tips_world)
            surf_local, d, _ = pq.on_surface(tips_local)                     # nearest point ON surface
            tgt_local[t, h] = surf_local
            dist[t, h] = d
            # back to world so the placement stage can target it directly (inverse of world->local)
            M = trimesh.transformations.quaternion_matrix(obj_quat[t])
            M[:3, 3] = obj_pos[t]
            tgt_world[t, h] = (M[:3, :3] @ surf_local.T).T + M[:3, 3]

    return tgt_world, tgt_local, dist < tau, dist


def report(contact: np.ndarray, dist: np.ndarray):
    """Summarize what was found so the targets can be sanity-checked before anything is moved."""
    for h, side in enumerate(HANDS):
        cf = contact[:, h].any(axis=1)                                       # frames this hand touches
        print(f"  {side:5s} hand: {cf.sum():3d}/{len(contact)} frames in contact")
        if cf.sum() == 0:
            continue
        for fi, f in enumerate(FINGERS):
            fc = contact[:, h, fi]
            if fc.any():
                print(f"      {f:7s} contact in {fc.sum():3d} frames, "
                      f"median gap {np.median(dist[fc, h, fi]) * 100:.1f} cm")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stageA_npz", help="Stage-A *_original.npz (for the object pose)")
    p.add_argument("object_obj", help="object mesh, e.g. models/table/table.obj")
    p.add_argument("handkp_npy", help="human hand keypoints (T,2,21,3)")
    p.add_argument("--tau", type=float, default=0.03, help="contact threshold, metres")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    handkp = np.load(args.handkp_npy)
    assert handkp.ndim == 4 and handkp.shape[1:] == (2, 21, 3), f"unexpected handkp {handkp.shape}"
    mesh = trimesh.load(args.object_obj, force="mesh")
    obj_pos, obj_quat = object_pose(args.stageA_npz)
    T = min(len(handkp), len(obj_pos))
    handkp, obj_pos, obj_quat = handkp[:T], obj_pos[:T], obj_quat[:T]

    tgt_world, tgt_local, contact, dist = fingertip_contacts(handkp, mesh, obj_pos, obj_quat, args.tau)
    report(contact, dist)

    out = args.out or os.path.splitext(os.path.basename(args.stageA_npz))[0].replace(
        "_original", "") + "_contact.npz"
    np.savez(out, tgt_world=tgt_world, tgt_local=tgt_local, contact=contact, dist=dist,
             tips=np.array(TIPS))
    print(f"[contact] {T} frames | tau {args.tau * 100:.0f} cm  ->  {out}")


if __name__ == "__main__":
    main()
