# HiPHI scaled-object comparison

This worktree provides the unchanged SMPL-X HOI and `object_scaled` implementation from the OMOMO comparison at `5d8320e`. The HiPHI runner, protocol and validation live in `../../../UMR-cmp-hiphi-object-frame/experiments/hiphi_scaled_object_frame_30fps_90/`. Outputs are in `../../../retarget_eval/hiphi_scaled_object_frame_30fps_90/`.

Local HiPHI meshes and scenes are copied from the existing world-scaled worktree and hashed in the output `local_assets.json`; no retargeting solver code is changed here.
