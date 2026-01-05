#!/bin/bash

###########################################################
# 1. Initialize conda (required for non-interactive shells)
###########################################################

# Change this to your own miniconda/anaconda path
source ~/anaconda3/etc/profile.d/conda.sh

# ===== Activate your environment =====
conda activate 14



###########################################################
# 2. Configurations
###########################################################

datasets=(0 1 2 3 4)
gpus=(0 1 2 3 4 5 6 7)

PY_FILE="00_14_template_for_all.py"   # your python script

###########################################################
# 3. Main loops
###########################################################

for dataset in "${datasets[@]}"; do

    echo "==========================================="
    echo "Running model: $model    dataset: $dataset"
    echo "==========================================="

    for gpu in "${gpus[@]}"; do

        echo "----> Running on GPU $gpu"

        python "$PY_FILE" \
            --d "$dataset" \
            --po "$gpu" &

        sleep 6
    done
    echo "Waiting for all GPU jobs  to finish..."
    wait
    echo "Finished ALL 8 GPUs for: $model + $dataset"
    echo ""

done


echo "==========================================="
echo "All jobs completed!"
echo "==========================================="
