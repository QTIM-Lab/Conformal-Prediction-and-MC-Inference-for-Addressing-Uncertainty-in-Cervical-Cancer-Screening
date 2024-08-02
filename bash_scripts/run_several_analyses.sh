#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Path to the Python script
PYTHON_SCRIPT_PATH="/sddata/projects/MICCAI_2024_UNSURE_Conformal-Prediction-and-MC-Inference-for-Addressing-Uncertainty-in-Cervical-Cancer-Screening_copy/src/full_avg_mc_pred_sets_pipeline.py"

# CSV files array (add your csv paths here)
PREDICTIONS_CSV_PATHS=(
    "/sddata/projects/MICCAI_2024_UNSURE_Conformal-Prediction-and-MC-Inference-for-Addressing-Uncertainty-in-Cervical-Cancer-Screening_copy/csvs/three_class_predictions.csv"
)

# Ground truth CSV path
GT_CSV_PATH="/sddata/projects/MICCAI_2024_UNSURE_Conformal-Prediction-and-MC-Inference-for-Addressing-Uncertainty-in-Cervical-Cancer-Screening_copy/csvs/three_class_ground_truths.csv"

# Dataset filter
DATASET_FILTER="test2"

# Parameters arrays
CUQ_TYPES=("lac" "aps")
ALPHAS=("0.05" "0.1" "0.2")
NUM_MC_COLS="50"
NUM_CLASSES="3"
LABEL_COL_NAME="CC_ST"

# Function to create a save path based on input parameters
generate_save_path() {
    local cuq_type=$1
    local alpha=$2
    local num_classes=$3
    local base_dir="/sddata/projects/MICCAI_2024_UNSURE_Conformal-Prediction-and-MC-Inference-for-Addressing-Uncertainty-in-Cervical-Cancer-Screening_copy/analysis"
    local alpha_str=$(printf "%.2f" "$alpha" | sed 's/\.//')

    echo "$base_dir/$num_classes/$cuq_type/alpha_$alpha_str"
}

# Iterate over each combination of parameters
for PREDICTIONS_CSV_PATH in "${PREDICTIONS_CSV_PATHS[@]}"; do
    for CUQ_TYPE in "${CUQ_TYPES[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            SAVE_PATH=$(generate_save_path "$CUQ_TYPE" "$ALPHA" "$NUM_CLASSES")
            echo "Running for CUQ_TYPE=$CUQ_TYPE, ALPHA=$ALPHA, SAVE_PATH=$SAVE_PATH"

            # Create the save directory if it doesn't exist
            mkdir -p "$SAVE_PATH"

            # Run the Python script with the current set of parameters
            python "$PYTHON_SCRIPT_PATH" \
                --predictions_csv_path "$PREDICTIONS_CSV_PATH" \
                --gt_csv_path "$GT_CSV_PATH" \
                --dataset_filter "$DATASET_FILTER" \
                --cuq_type "$CUQ_TYPE" \
                --alpha "$ALPHA" \
                --cuq_df_save_dir "$SAVE_PATH" \
                --num_mc_cols "$NUM_MC_COLS" \
                --num_classes "$NUM_CLASSES" \
                --label_col_name "$LABEL_COL_NAME" \
                --save_path "$SAVE_PATH"
        done
    done
done
