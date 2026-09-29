# OMOMO world-scaled objects

Branch: comparison/omomo-scaled-world-30fps.

Use `--task-config.object-scale-mode world_scaled` with the existing SMPL-X HOI input format. It scales collision geometry, interaction points, and all three object translation coordinates by the human scale. The translation receives the same floor-origin adjustment as the human. Object rotations and robot dimensions remain unchanged. The default `original` mode retains the earlier behavior.

The G1 object scenes in this branch use the six native convex hull meshes, restored from the existing holosoma_hull copies. This retains the optimization geometry used in the previous primary comparison. Shared CoACD geometry is used separately for evaluation.

The frozen cohort, CPU runner, geometry tests, evaluator, and protocol are in the sibling UMR-cmp-omomo-scaled worktree under `experiments/omomo_scaled_world_30fps_90`. Local asset hashes are recorded there. The source dataset and model assets are not uploaded.

Results update automatically in `retarget_eval/omomo_scaled_world_30fps_90/REPORT.md`, with JSON/CSV companions. No solver weights or constraint tolerances are changed.
