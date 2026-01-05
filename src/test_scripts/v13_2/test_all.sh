#!/bin/bash
# ==============================
# Multi-GPU Sequential Job Runner
# ==============================

# ----- Initialize conda -----
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 14    # <--- your conda environment

# ----- GPU list -----
GPUS=(0 1 2 3 4 5 6 7)

# ----- Python scripts to run in sequence -----
SCRIPTS=(
    "05_cot_reward_livew_v0.py"
    "04_cot_reward_agiqa_v0.py"
    "03_cot_reward_csiq_v0.py"
    "02_cot_reward_kadid_v0.py"
    "01_cot_reward_spaq_v0.py"
)

# ----- Delay between GPU launches (optional) -----
DELAY=10   # seconds

# ===============================
# Main loop: run scripts in order
# ===============================

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "==============================================="
    echo "Starting script: $SCRIPT"
    echo "Running jobs on GPUs: ${GPUS[*]}"
    echo "==============================================="
    
    # Launch one job per GPU
    for GPU in "${GPUS[@]}"; do
        echo "[GPU $GPU] Launching: python $SCRIPT --po $GPU"
        
        python "$SCRIPT" --po $GPU  &

        sleep $DELAY
    done

    echo "Waiting for all GPU jobs for $SCRIPT to finish..."
    wait   # <-- waits for all background GPU jobs
    echo "Completed all jobs for script: $SCRIPT"
    echo
done

echo "================================================="
echo "ALL SCRIPTS COMPLETED SUCCESSFULLY!"
echo "================================================="
