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

### End-to-end: reproduce the Wuji dexterous grasp (verified, fully reproducible)

Produces the dexterous-grasp result for `sub3_largebox_003` and renders an mp4.
The body+box solve is done in holosoma (`omniretarget` env); the *finger*
optimization is done by the external [`wuji-retargeting`](https://github.com/wuji-technology/wuji-retargeting)
tool (`phase2_dex.py`, run in its own `pinocchio` env — here called `wujiret`).
The retargeter is deterministic (`np.random.seed(0)`), so re-runs are bit-identical.

**Prerequisites:** the external `wuji-hand-description` + `wuji-retargeting` repos,
and a conda env with `pinocchio` for the finger step (see `wuji-retargeting`'s README).

```bash
# ── STEP 0 — build the Wuji models  (omniretarget env; needs wuji-hand-description)
conda activate omniretarget
cd src/holosoma_retargeting/holosoma_retargeting
export WUJI_HAND_DESCRIPTION=~/Downloads/wuji-hand-description   # default: ~/wuji-hand-description
python wuji/make_wuji_model.py largebox        # -> models/g1/g1_29dof_wuji_w_largebox.xml        (nq 83, render model)
python wuji/make_wuji_model.py largebox --weld # -> models/g1/g1_29dof_wuji_welded_w_largebox.xml (nq 43, Step 1 model)
#   (the models are committed and work as-is; run Step 0 only to regenerate them.
#    The "Attach conflict" warnings are harmless — MuJoCo just keeps the G1's sim settings.)

# ── STEP 1 — welded body retarget  -> demo_results_wuji_welded  (43-DOF: body + box)
#    The robot-urdf-file name only selects the MuJoCo xml (…_w_largebox.xml); the .urdf
#    itself is never loaded headless.  Welded model has no finger DOF -> smplh_wuji_body.
python examples/robot_retarget.py --data-format smplh_wuji_body \
    --robot-config.robot-urdf-file models/g1/g1_29dof_wuji_welded.urdf \
    --save-dir demo_results_wuji_welded
#   -> demo_results_wuji_welded/sub3_largebox_003_original.npz   (196, 43)

# ── STEP 2 — provide human_joints for phase2_dex (it computes the fingers itself and only
#    needs the human keypoints, which the welded run already saved). Just a copy:
mkdir -p demo_results_wuji
cp demo_results_wuji_welded/sub3_largebox_003_original.npz \
   demo_results_wuji/sub3_largebox_003.npz

# ── STEP 3 — dex fingers + merge -> demo_results_wuji_dex   (external wuji-retargeting, pinocchio env)
conda activate wujiret
cd ~/Downloads/wuji-retargeting
python phase2_dex.py
#   -> …/holosoma/…/demo_results_wuji_dex/sub3_largebox_003.npz   (196, 83)
#   NB: phase2_dex.py has the holosoma path hardcoded (HOLO=...) and reads the fixed dir
#       names demo_results_wuji_welded/ and demo_results_wuji/ — edit those if you relocate.

# ── STEP 4 — render the mp4  (omniretarget env)
conda activate omniretarget
cd ~/Downloads/holosoma/src/holosoma_retargeting/holosoma_retargeting
python wuji/render_mp4.py demo_results_wuji_dex/sub3_largebox_003.npz out_dex.mp4 --ghost
python wuji/play_wuji.py  demo_results_wuji_dex/sub3_largebox_003.npz 8082   # or view interactively
```

> The built `models/g1/g1_29dof_wuji*.xml` work out-of-the-box; the
> `wuji-hand-description` repo is only needed to *regenerate* them (Step 0).

**Holosoma-only variant (no external tool, lower-quality fingers):** the in-repo
`smplh_wuji` all-in-one path retargets body + fingers + box in one solve — but the
finger geoms vs. the box make it fragile (the fingers can penetrate). Use it only
if you can't run the external dex step:
```bash
python examples/robot_retarget.py --data-format smplh_wuji \
    --robot-config.robot-urdf-file models/g1/g1_29dof_wuji.urdf \
    --save-dir demo_results_wuji_obj            # -> 83-DOF, all-in-one
```

---

## 2. HUMOTO interaction-mesh retargeting

**New / changed files**
| File | Purpose |
|---|---|
| `src/interaction_mesh_retargeter.py` (+164) | HUMOTO retargeting logic |
| `src/utils.py` (+30) | helper additions |
| `demo_data/height_dict.pkl` | per-subject heights (used for the human→robot scale) |
| `viser_g1.py` | interactive viser viewer for a retargeted G1 + object |
| `render_g1_mp4.py` | MP4 render of a retargeted G1 + object |
| `models/g1/g1_29dof_w_<object>.xml` | 26 per-object G1 models (hammer, knife, spatula, guitar, floor_lamp, …) |

**Run**:
```bash
cd src/holosoma_retargeting/holosoma_retargeting
python viser_g1.py     <task_name> <object_name> [save_dir] [--port 8080]
python render_g1_mp4.py <task_name> <object_name> [save_dir] [out_dir]
# e.g. python viser_g1.py sub14_suitcase_001 suitcase demo_results/g1/object_interaction/omomo_test
```

---

## External dependencies (not committed)

Like the upstream repo, motion **input data** and **object meshes** are *not* in
git — fetch them from the datasets and drop them in place:

- **Object meshes** — the per-object models `models/g1/g1_29dof_w_<obj>.xml`
  reference `models/<obj>/<obj>.obj` (e.g. `models/hammer/hammer.obj`). Only
  `models/largebox/` ships in-repo; the other ~26 object folders come from the
  **HUMOTO / InterAct** dataset. Place them under `models/<obj>/`.
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
