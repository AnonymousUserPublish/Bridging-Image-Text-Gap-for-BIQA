#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

# ===== Activate your environment =====
conda activate 14

GPUS=(0 1 2 3 4 5 6 7)

for GPU in "${GPUS[@]}"; do
    echo "Launching job on GPU $GPU"
    python 00_cot_reward_koniq_v0.py --po $GPU &

    echo "Sleep 10s before launching next job..."
    sleep 10
done

wait
echo "All jobs finished."

