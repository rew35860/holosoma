# OMOMO largebox: scaled mesh with two object-height rules

Sequence `sub10_largebox_061`, 117 frames at 30 FPS. This is a paired visual check, not a new benchmark cohort. Outputs and measurements are in `retarget_eval/omomo_largebox_scaled_frames`.

Both runs use the same source SMPL-X/HOI input, robot, solver settings, body target, object mesh, mesh scale `s=0.802398`, object X/Y and orientation. The world result is the existing validated OmniRetarget output from `retarget_eval/omomo_scaled_world_30fps_90`. The second run was solved fresh on branch `comparison/omomo-largebox-scaling-frames`, based on the same tested OmniRetarget commit `dfd1c2f`. Its `object_scaled` option uses the **same mesh and object contact points as the world run**, with the original OmniRetarget object-height preprocessing. It adds an option in three small code changes; no objective, solver tolerance, G1 model or SMPL-X input is altered. For this clip, the original preprocessing takes its floor-rest branch.

Let `v` be a local mesh vertex, `R` object orientation, `t` source translation, `z₀=t_z[0]=0.18461958 m`, and `z_min=0.030947864 m` the native human toe-floor correction. Both methods use mesh vertex `s v` and object X/Y translation `s t_xy`.

- **World:** `t'_z=s(t_z-z_min)`. The same floor-offset and stature transform is applied to human and object.
- **Object frame / native height:** `t'_z=z₀+s(t_z-z₀)`. It scales vertical movement about the object’s initial translation, preserving `z₀`.

The object-frame run therefore places the entire object `61.314 mm` above the world-run target throughout this clip. This does not by itself tell whether either object's bottom contacts the floor; that depends on the mesh-local origin and the source object pose.

## Measured scene clearance

`renders/floor_clearance.json` contains per-frame signed distances. `renders/summary.json` contains first-frame and median values. The object number is the lowest compiled visual-mesh vertex in world coordinates, relative to floor `z=0`. Robot feet use MuJoCo `mj_geomDistance` between the ground plane and all collision geoms on each ankle-roll foot. Positive is above the floor, negative is penetration.

| Mode | Object bottom, first frame | Left foot, first frame | Right foot, first frame | Median lower foot |
|---|---:|---:|---:|---:|
| World | -26.2 mm | -1.5 mm | -1.5 mm | -1.5 mm |
| Object-frame / native Z | +35.1 mm | -1.4 mm | -1.4 mm | -1.5 mm |

Neither version has both feet more than 10 mm above the floor in any frame. The lifted foot during walking is a swing foot, while the other remains near the ground. The largebox world run sinks the box about 26 mm, and the object-frame run leaves it about 35 mm above the floor. The 30.95 mm human floor offset is a material part of why the uniformly transformed world result sits below ground here; world scaling is not inherently equivalent to ground penetration. Scaling around world `z=0` would give a different placement.

The earlier HUMOTO side-table comparison used **full-size meshes**, with native and matched object-height rules, not these two scaled-mesh runs. Its saved `carry_side_table_with_both_hands_walk_around-536` outputs place the native object about 21.5 mm lower than the matched target. On its existing evaluation scene, at least one foot is within 10 mm of the floor in every frame for both variants. Thus that particular saved pair does not show a whole-body floating-foot problem. A raised foot in the saved mid-frame is the walking/swing foot. This observation does not establish which separate clip or render a remembered artifact came from.

## Reproduce and inspect

`launch.sh` runs one local OmniRetarget solve if needed and renders the two outputs. It runs CPU-only with 75% CPU quota, 4 GiB maximum memory, low process priority and no GPU. The existing HiPHI batch and policy-training GPU process are left running. `run_compare.py` validates output shape, identical human target, object X/Y/rotation, and the predicted object Z difference. `render_compare.py` renders with Mesa llvmpipe, reuses the exact source and world-result panels from the previously validated video, draws the new object-frame panel with the same camera rule, and decodes all three MP4 files to confirm 117 frames at 30 FPS. The combined MP4, solo MP4s, first and mid-frame PNGs, contact sheet and gallery are in `renders/`.
