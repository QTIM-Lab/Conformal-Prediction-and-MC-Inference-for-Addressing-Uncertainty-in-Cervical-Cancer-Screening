This repo serves as the reproducibility for the project Conformal Prediction and Monte Carlo Inference for Addressing Uncertainty in Cervical Cancer Screening for the MICCAI 2024 UNSURE Workshop.

To run this project from scratch, please run MICCAI_UNSURE_Workshop_Anon_Git/bash_scripts
/run_several_analyses.sh with the following changes and correct the paths in `PYTHON_SCRIPT_PATH`, `PREDICTIONS_CSV_PATHS`, `GT_CSV_PATH`, and choose where to send the analysis in `local base_dir`.

Now, to run the three-class version, please use:

`PREDICTIONS_CSV_PATHS="/path/to/three_class_predictions.csv"`
`GT_CSV_PATH="/path/to/three_class_ground_truths.csv"`
`DATASET_FILTER="test2`

The output will be in the `local_base_dir`.

Now, to run the two-class version, please use:

`PREDICTIONS_CSV_PATHS="/path/to/binary_predictions_with_ground_truths.csv"`
`GT_CSV_PATH="/path/to/binary_predictions_with_ground_truths.csv"`
`DATASET_FILTER="test`

To generate the Gray Zone comparison csv, please run:

`python3 src/two_and_three_class_comparison.py` with the appropriate `cuq_df_3c_path`, `cuq_df_2c_path` and `save_path`.