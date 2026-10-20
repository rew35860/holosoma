"""Additional robot-comfort diagnostics; frozen contact/collision metrics stay intact."""
import sys
import numpy as np
import mujoco
from config import *
sys.path.insert(0, str(RET/'retarget_pipeline'))
from pipeline_v2 import CoupledRefiner, Settings


def main(variant, seq):
    dest = OUT/'variants'/variant/'clips'/seq
    info = inputs(seq)
    model = mujoco.MjModel.from_xml_path(str(info['model']))
    ref = CoupledRefiner(model, info['object'], Settings(**settings(variant)))
    q = np.load(dest/'final_v2.npz')['qpos']
    base = np.load(PILOT/'clips'/seq/'final_v2.npz')['qpos']
    mask = np.load(dest/'per_frame_metrics.npz')['source_contact'].any(-1)
    arm = np.array(ref.arm_columns)
    a = q[:, ref.qadr[arm]]
    lo, hi = ref.lo[arm], ref.hi[arm]
    margin = np.minimum(a-lo, hi-a)
    minimum = margin.min(-1)
    normalized = (margin/(hi-lo)).min(-1)
    change = np.degrees(a-base[:, ref.qadr[arm]])
    # Only derivative stencils entirely within contact enter contact-only summaries.
    result = dict(sequence=seq, variant=variant, contact_hand_frames=int(mask.sum()),
        contact_arm_min_margin_mean_deg=float(np.degrees(minimum[mask]).mean()),
        contact_arm_min_margin_p05_deg=float(np.percentile(np.degrees(minimum[mask]), 5)),
        contact_arm_min_margin_min_deg=float(np.degrees(minimum[mask]).min()),
        contact_hand_near_limit_5deg_pct=float(100*np.mean(minimum[mask] < np.deg2rad(5))),
        contact_arm_min_normalized_margin_mean_pct=float(100*normalized[mask].mean()),
        arm_change_from_full_rms_deg=float(np.sqrt(np.mean(change[mask]**2))),
        arm_change_from_full_max_deg=float(np.abs(change).max()),
        definitions='One sample per contacting hand-frame; minimum over seven arm/wrist joints. Derivatives are forward differences at 30 Hz, no filtering. These are kinematic comfort proxies, not human imitation scores.')
    for order, label in ((1,'velocity'), (2,'acceleration'), (3,'jerk')):
        values = np.diff(a, n=order, axis=0)*30**order
        select = np.logical_and.reduce([mask[i:len(mask)-order+i] for i in range(order+1)])
        result[f'contact_arm_{label}_rms_rad_s{order}'] = float(np.sqrt(np.mean(values[select]**2)))
        result[f'contact_arm_{label}_max_rad_s{order}'] = float(np.abs(values[select]).max())
    per_joint=[]
    for side in range(2):
        for j,column in enumerate(arm[side]):
            values=np.degrees(margin[:,side,j][mask[:,side]])
            per_joint.append(dict(joint=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,ref.joints[column]),
                contact_frames=len(values),margin_mean_deg=float(values.mean()) if len(values) else None,
                margin_min_deg=float(values.min()) if len(values) else None,
                near_limit_5deg_pct=float(100*np.mean(values<5)) if len(values) else None,
                max_change_from_full_deg=float(np.abs(change[:,side,j]).max())))
    write(dest/'comfort_per_joint.json',per_joint)
    np.savez_compressed(dest/'comfort_per_frame.npz', contact_hand=mask,
        arm_min_margin_rad=minimum, arm_min_normalized_margin=normalized,
        arm_change_from_full_deg=change)
    write(dest/'comfort.json', result)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
