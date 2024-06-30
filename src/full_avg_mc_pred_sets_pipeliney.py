import argparse
import os
import sys
import pandas as pd
from create_cuq_dfs import additions_to_csvs, create_cuq_df, create_mc_df
from mc_cervix_analysis_cov_std_dev import analyse_cuq_df_two_class

def main(predictions_csv_path, gt_csv_path, dataset_filter, cuq_type, alpha, cuq_df_save_dir, num_mc_cols, num_classes, label_col_name, save_path):

    # Ensure the save directories exist
    os.makedirs(cuq_df_save_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Load prediction and ground truth data
    predictions_df = pd.read_csv(predictions_csv_path)
    gt_df = pd.read_csv(gt_csv_path)
    gt_df_subset = gt_df[gt_df['dataset'] == dataset_filter]

    # Process data
    predictions_df = additions_to_csvs(predictions_df, gt_df_subset)
    cuq_df_eval = create_cuq_df(cuq_type, predictions_df, label_col_name, 'pred', 'predicted_class', 'MASKED_IMG_ID', alpha, 0.2, True, num_mc_cols, num_classes)
    cuq_df = create_mc_df(cuq_type, predictions_df, cuq_df_eval, label_col_name, alpha, num_mc_cols, num_classes, cuq_df_save_dir)

    # Analysis
    if num_classes == 2:
        print('We are running a two-class analysis')
        analyse_cuq_df_two_class(cuq_df, 'mc', save_path, os.path.basename(save_path))
    elif num_classes == 3:
        print('We are running a three-class analysis')
        analyse_cuq_df_two_class(cuq_df, 'mc', save_path, os.path.basename(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CUQ Analysis")
    parser.add_argument('-p', '--predictions_csv_path', type=str, required=True, help="Path to predictions CSV")
    parser.add_argument('-g', '--gt_csv_path', type=str, required=True, help="Path to ground truth CSV")
    parser.add_argument('-d', '--dataset_filter', type=str, required=True, help="Dataset filter (e.g., 'test2')")
    parser.add_argument('-t', '--cuq_type', type=str, default='lac', help="CUQ type")
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help="Alpha value")
    parser.add_argument('-s', '--cuq_df_save_dir', type=str, required=True, help="Directory to save CUQ dataframe")
    parser.add_argument('-m', '--num_mc_cols', type=int, default=50, help="Number of Monte Carlo columns")
    parser.add_argument('-c', '--num_classes', type=int, default=3, help="Number of classes")
    parser.add_argument('-l', '--label_col_name', type=str, required=True, help="Label column name")
    parser.add_argument('-o', '--save_path', type=str, required=True, help="Path to save analysis results")

    args = parser.parse_args()
    print(args)
    # sys.exit()
    main(args.predictions_csv_path, args.gt_csv_path, args.dataset_filter, args.cuq_type, args.alpha, args.cuq_df_save_dir, args.num_mc_cols, args.num_classes, args.label_col_name, args.save_path)
