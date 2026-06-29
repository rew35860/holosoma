#!/usr/bin/env python
"""Phase 2 merge: body+arms+object from the welded OmniRetarget run, finger
joints from the dexterous finger-tracking run -> one full-DOF qpos on the
non-welded Wuji model (with the box). Matched by joint name.

Body source  : demo_results_wuji_welded/sub3_largebox_003_original.npz  (43-dim, body+box)
Finger source: demo_results_wuji/sub3_largebox_003.npz                  (76-dim, fingers track human)
Output       : demo_results_wuji_merged/sub3_largebox_003.npz           (83-dim, on g1_29dof_wuji_w_largebox.xml)
"""
from pathlib import Path
import numpy as np
import mujoco

REPO = Path(__file__).resolve().parent
G1 = REPO / "models/g1"
M_out = mujoco.MjModel.from_xml_path(str(G1 / "g1_29dof_wuji_w_largebox.xml"))   # 83 dof target
M_body = mujoco.MjModel.from_xml_path(str(G1 / "g1_29dof_wuji_welded_w_largebox.xml"))  # 43
M_fing = mujoco.MjModel.from_xml_path(str(G1 / "g1_29dof_wuji.xml"))             # 76

q_body = np.load(REPO / "demo_results_wuji_welded/sub3_largebox_003_original.npz", allow_pickle=True)["qpos"]
q_fing = np.load(REPO / "demo_results_wuji/sub3_largebox_003.npz", allow_pickle=True)["qpos"]
T = q_body.shape[0]
assert q_fing.shape[0] == T


def addr(m, name):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(m.jnt_qposadr[j]) if j >= 0 else None


merged = np.zeros((T, M_out.nq), dtype=np.float64)
# base (free joint, qpos 0:7) and object (last free joint) come from the body run
merged[:, 0:7] = q_body[:, 0:7]
merged[:, M_out.nq - 7:] = q_body[:, M_body.nq - 7:]   # object 7-dof

n_fing = n_body = 0
for j in range(M_out.njnt):
    name = mujoco.mj_id2name(M_out, mujoco.mjtObj.mjOBJ_JOINT, j)
    if not name:                      # the two unnamed free joints (base/object) handled above
        continue
    a_out = int(M_out.jnt_qposadr[j])
    if "finger" in name:
        a = addr(M_fing, name); merged[:, a_out] = q_fing[:, a]; n_fing += 1
    else:                             # G1 body/arm joint
        a = addr(M_body, name); merged[:, a_out] = q_body[:, a]; n_body += 1

out = REPO / "demo_results_wuji_merged"
out.mkdir(exist_ok=True)
np.savez(out / "sub3_largebox_003.npz", qpos=merged,
         human_joints=np.load(REPO / "demo_results_wuji/sub3_largebox_003.npz", allow_pickle=True)["human_joints"],
         fps=30)
print(f"[merge] body joints from welded: {n_body} | finger joints from tracker: {n_fing}")
print(f"[merge] wrote demo_results_wuji_merged/sub3_largebox_003.npz  qpos {merged.shape}")
