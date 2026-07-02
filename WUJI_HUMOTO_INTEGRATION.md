# Wuji-hand & HUMOTO integration (fork notes)

This branch (`feature/wuji-hand-humoto`) layers two pieces of work on top of
upstream `amazon-far/holosoma` (`main` @ `80f1221`):

1. **Wuji dexterous-hand retargeting** for the Unitree G1 (20-DOF hand).
2. **HUMOTO interaction-mesh retargeting** + per-object G1 models.

The two changesets are almost entirely in different files; the only shared file
is the retargeter source, and they were merged without conflict. All work lives
under `src/holosoma_retargeting/holosoma_retargeting/`.

---

## 1. Wuji dexterous-hand retargeting

Grafts the 20-DOF Wuji hand onto the G1 and retargets human (SMPLH/MANO) hand
motion onto it, on top of OmniRetarget's body/object solve.

**New / changed files**
| File | Purpose |
|---|---|
| `wuji/make_wuji_model.py` | Generate the G1+Wuji MJCF via `MjSpec.attach` (optional finger **weld** + object **baking**) → `models/g1/g1_29dof_wuji[_welded][_w_<obj>].xml` |
| `config_types/data_type.py` | New data formats: `smplh_wuji`, `smplh_wuji_body` (10 fingertips → Wuji finger links) |
| `examples/robot_retarget.py` | Loader + object-interaction support for the new SMPLH-hand formats |
| `wuji/merge_wuji.py` | Splice a welded-body solve + a finger solve into one full-DOF qpos (holosoma-only finger variant) |
| `wuji/play_wuji.py` | Viser playback of a result (MuJoCo FK, no URDF) |
| `wuji/render_mp4.py` | Offscreen MuJoCo render of a qpos trajectory (`--ghost`, `--track`) |
| `models/g1/g1_29dof_wuji*.xml` + `models/g1/assets/*.STL` | Built Wuji-hand models + meshes (committed, ready to run) |

### End-to-end: reproduce the Wuji dexterous grasp (verified)

`sub3_largebox_003` → dexterous-grasp npz + mp4. Body+box solve in holosoma
(`omniretarget`); fingers via the external
[`wuji-retargeting`](https://github.com/wuji-technology/wuji-retargeting) `phase2_dex.py`
(its own `pinocchio` env, here `wujiret`). Deterministic (`np.random.seed(0)`) → bit-identical.

**Prereqs:** the `wuji-hand-description` + `wuji-retargeting` repos. Steps run in one
shell; `conda activate` / `cd` shown only when the env or dir changes.

**Step 0 — build the Wuji models.** Already committed; run only to regenerate.
```bash
conda activate omniretarget
cd src/holosoma_retargeting/holosoma_retargeting
export WUJI_HAND_DESCRIPTION=~/Downloads/wuji-hand-description   # default: ~/wuji-hand-description
python wuji/make_wuji_model.py largebox         # g1_29dof_wuji_w_largebox.xml         (nq 83, render)
python wuji/make_wuji_model.py largebox --weld  # g1_29dof_wuji_welded_w_largebox.xml  (nq 43, Step 1)
```

**Step 1 — welded body retarget** → 43-DOF (body + box).
```bash
python examples/robot_retarget.py --data-format smplh_wuji_body \
    --robot-config.robot-urdf-file models/g1/g1_29dof_wuji_welded.urdf \
    --save-dir demo_results_wuji_welded
# -> demo_results_wuji_welded/sub3_largebox_003_original.npz   (196, 43)
```

**Step 2 — stage human joints for `phase2_dex`** (a copy; it recomputes fingers itself).
```bash
mkdir -p demo_results_wuji
cp demo_results_wuji_welded/sub3_largebox_003_original.npz demo_results_wuji/sub3_largebox_003.npz
```

**Step 3 — dex fingers + merge** → 83-DOF.
```bash
conda activate wujiret
cd ~/Downloads/wuji-retargeting
python phase2_dex.py
# -> holosoma/.../demo_results_wuji_dex/sub3_largebox_003.npz   (196, 83)
```

**Step 4 — render / view.**
```bash
conda activate omniretarget
cd ~/Downloads/holosoma/src/holosoma_retargeting/holosoma_retargeting
python wuji/render_mp4.py demo_results_wuji_dex/sub3_largebox_003.npz out_dex.mp4 --ghost
python wuji/play_wuji.py  demo_results_wuji_dex/sub3_largebox_003.npz 8082   # interactive
```

Notes:
- `--robot-urdf-file` only *names* the scene xml (`…_w_largebox.xml`); the `.urdf` is never loaded headless.
- `phase2_dex.py` hardcodes the holosoma path + the `demo_results_wuji_welded/` / `demo_results_wuji/` names — edit if you relocate.
- Step 0's "Attach conflict" warnings are harmless (MuJoCo keeps the G1's sim settings).

**Holosoma-only variant** — no external tool; fingers can penetrate the box, so use only if you can't run Step 3:
```bash
python examples/robot_retarget.py --data-format smplh_wuji \
    --robot-config.robot-urdf-file models/g1/g1_29dof_wuji.urdf \
    --save-dir demo_results_wuji_obj            # 83-DOF, all-in-one
```

---

## 2. HUMOTO interaction-mesh retargeting

**New / changed files**
| File | Purpose |
|---|---|
| `src/interaction_mesh_retargeter.py` | HUMOTO retargeting logic (interaction-mesh QP solve) |
| `src/diagnostics.py` | solver-failure diagnostics (`diagnose_infeasibility`) — split out of the retargeter so it only runs on a failed CVXPY solve |
| `src/utils.py` | helper additions |
| `demo_data/height_dict.pkl` | per-subject heights (used for the human→robot scale) |
| `viser_g1.py` | interactive viser viewer for a retargeted G1 + object |
| `render_g1_mp4.py` | MP4 render of a retargeted G1 + object |
| `models/g1/g1_29dof_w_<object>.xml` | 26 per-object G1 models (hammer, knife, spatula, guitar, floor_lamp, …) |

### End-to-end: reproduce a HUMOTO sequence (verified)

Raw Mixamo FBX → up_bone pkl → direct `.pt` → G1 npz → mp4, across three envs.
Example `checking_organizer_medium_on_table-289` / `organizer_medium`. The direct
`.pt` drops Mixamo joint positions into the 52 SMPLH slots (no SMPL-X fit);
OmniRetarget reads only those + the object pose. Set once:
```bash
SEQ=checking_organizer_medium_on_table-289 ; OBJ=organizer_medium
```

**Step 0 — raw FBX → up_bone pkl.** Env `humoto` (python 3.10, bpy==4.0.0; see humoto/README).
```bash
conda activate humoto
cd ~/Downloads/humoto/scripts
RAW=~/Downloads/humoto/humoto/humoto_0805
python clear_human_scale.py    -d $RAW/$SEQ          -o /tmp/h_scale
python transfer_human_model.py -d /tmp/h_scale/$SEQ  -m ../human_model/human_model_without_texture_up_bone.fbx -o /tmp/h_upbone
python extract_pk_data.py      -d /tmp/h_upbone/$SEQ -o ~/Downloads/humoto_data/humoto_upbone_pkl
# -> humoto_upbone_pkl/$SEQ/$SEQ.pkl   (no -m: objects come from humoto_objects_0805)
```

**Step 1 — up_bone pkl → direct .pt.** Env `interact` (torch, smplx, trimesh, scipy, tqdm).
```bash
conda activate interact
cd ~/Downloads/InterAct/simulation
export HUMOTO_UPBONE=~/Downloads/humoto_data/humoto_upbone_pkl
export HUMOTO_REPO=~/Downloads/humoto
export HUMOTO_OBJECTS=~/Downloads/humoto/humoto/humoto_objects_0805
python humoto_direct_to_pt.py $SEQ $OBJ
# -> InterAct/result/humoto_pt/$SEQ.pt   (T, 591)
```

**Step 2 — .pt → retargeted G1 npz.** Env `omniretarget`.
```bash
conda activate omniretarget
cd ~/Downloads/holosoma/src/holosoma_retargeting/holosoma_retargeting
mkdir -p models/$OBJ                                                 # mount object mesh (gitignored symlink)
ln -sf ~/Downloads/humoto/humoto/humoto_objects_0805/$OBJ/$OBJ.obj  models/$OBJ/$OBJ.obj
python examples/robot_retarget.py \
  --task-type object_interaction --data-format smplh \
  --task-name $SEQ --data-path ~/Downloads/InterAct/result/humoto_pt \
  --task-config.object-name $OBJ \
  --robot-config.robot-urdf-file models/g1/g1_29dof.urdf \
  --save-dir demo_results/g1/object_interaction/humoto
# -> .../object_interaction/humoto/${SEQ}_original.npz   (T, 43)
#    (--robot-urdf-file only NAMES the scene xml: g1_29dof.urdf -> g1_29dof_w_$OBJ.xml)
```

**Step 3 — render / view.**
```bash
python render_g1_mp4.py $SEQ $OBJ demo_results/g1/object_interaction/humoto
python viser_g1.py      $SEQ $OBJ demo_results/g1/object_interaction/humoto [--port 8080]
```

---

## External dependencies (not committed)

Like the upstream repo, motion **input data** and **object meshes** are *not* in
git — fetch them from the datasets and drop them in place:

- **Object meshes** — the per-object models `models/g1/g1_29dof_w_<obj>.xml`
  reference `models/<obj>/<obj>.obj` (e.g. `models/hammer/hammer.obj`). Only
  `models/largebox/largebox.obj` ships in-repo; every other `models/*/*.obj` is
  **gitignored** and mounted as a symlink into the HUMOTO dataset — never committed:
  ```bash
  ln -s ~/Downloads/humoto/humoto/humoto_objects_0805/<obj>/<obj>.obj  models/<obj>/<obj>.obj
  ```
- **`wuji-hand-description`** — only to regenerate the Wuji models (Step 0).
- **`wuji-retargeting`** ([wuji-technology/wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting)) —
  the external finger optimizer used by Step 3 (`phase2_dex.py`). Needs its own
  `pinocchio` conda env. Not part of this repo by design; documented above.
- **Motion datasets** — OMOMO / HUMOTO inputs, as in upstream.

## Gitignored (regenerable, not shared)
`demo_results_*/` (retarget outputs), `renders_*/` / `video_results/`,
`*.mp4`, `*_sample.png`, `MUJOCO_LOG.TXT`.

## Notes
- Dropped from the HUMOTO tree as noise: executable-bit/CRLF-only churn on ~20
  setup/CI shell scripts, and a cosmetic `robot_retarget.py` default-output
  change (`omomo`→`humoto`) — upstream default kept since this branch serves
  both datasets.
