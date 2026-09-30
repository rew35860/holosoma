#!/usr/bin/env bash
set -euo pipefail
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=/home/natcha/Downloads/retarget_eval/omomo_largebox_scaled_frames
mkdir -p "$OUT"
systemd-run --user --unit=omomo-largebox-scaling-frames \
 --property=MemoryMax=4G --property=MemoryHigh=3G --property=MemorySwapMax=0 \
 --property=CPUQuota=75% --property=CPUWeight=1 --property=IOWeight=1 \
 --property=Nice=19 --property=OOMPolicy=stop \
 --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=1 \
 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
 --property="StandardOutput=append:$OUT/run.log" \
 --property="StandardError=append:$OUT/run.log" \
 /home/natcha/miniconda3/envs/omniretarget/bin/python "$CODE/run_compare.py"
