#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$CODE_DIR/../../.." && pwd)"
OUT_DIR="$ROOT_DIR/retarget_eval/omomo_scaled_object_frame_30fps_90"
PY="$HOME/miniconda3/envs/omniretarget/bin/python"
"$PY" "$CODE_DIR/prepare.py"
systemd-run --user --unit=omomo-scaled-object-frame-90 --collect \
  --property=CPUQuota=75% --property=MemoryMax=4G --property=MemoryHigh=3G \
  --property=MemorySwapMax=0 --property=Nice=19 --property=CPUWeight=1 \
  --property=IOWeight=1 \
  --property="StandardOutput=append:$OUT_DIR/batch_run.log" \
  --property="StandardError=append:$OUT_DIR/batch_run.log" \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=NVIDIA_VISIBLE_DEVICES=none \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=RAYON_NUM_THREADS=1 --setenv=MUJOCO_GL=disable \
  --setenv=PYTHONUNBUFFERED=1 \
  --working-directory="$ROOT_DIR/holosoma-cmp-omomo-object-frame" \
  "$PY" "$CODE_DIR/run_batch.py"
