# HiPHI world-scaled comparison

Branch `comparison/hiphi-scaled-world-30fps` preserves the tested SMPL-X HOI and world-scaling implementation from `dfd1c2f`. No additional solver changes are needed for the converted HiPHI inputs.

The experiment runner, protocol, fixed 90-clip cohort and asset hashes are in the matching UMR worktree: `UMR-cmp-hiphi-scaled/experiments/hiphi_scaled_world_30fps_90`. Results are in `retarget_eval/hiphi_scaled_world_30fps_90`.

Native convex-hull scene XMLs and metre-scale HiPHI object meshes are imported locally from `holosoma_cmp_hiphi`, verified by SHA-256, and remain outside Git. Source SMPL-X comes from the historical official hiphi2smplx body-only conversion, without later hand/shape edits. The active training repository is not imported.
