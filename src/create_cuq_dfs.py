#########
# Imports
#########

import numpy as np
import pandas as pd
import os
import sys
import torch
import ast
from sklearn.model_selection import train_test_split
from cuq_classification_algorithms import *

###########
# Functions
###########

def additions_to_csvs(df, gt_df):
    mc_cols = [col for col in df.columns if 'mc_' in col][:-2]  # -2 is for the pred_mc and soft_pred_mc

    average_mc_expected_values = []
    rounded_average_mc_expected_values = []
    standard_deviations_mc_expected_values = []
    coefficients_of_variation_mc_expected_values = []  # New list for storing coefficients of variation

    for index, row in df.iterrows():
        mc_expected_values = []
        for col in mc_cols:
            mc_probs = ast.literal_eval(row[col])
            if len(mc_probs) == 3:
                mc_expected_values.append(0*mc_probs[0] + 1*mc_probs[1] + 2*mc_probs[2])
            elif len(mc_probs) == 2:
                mc_expected_values.append(0*mc_probs[0] + 1*mc_probs[1])

        average_mc_expected_value = np.average(mc_expected_values)
        standard_deviation_mc_expected_value = np.std(mc_expected_values)
        coefficient_of_variation_mc_expected_value = (standard_deviation_mc_expected_value / average_mc_expected_value) * 100

        average_mc_expected_values.append(average_mc_expected_value)
        rounded_average_mc_expected_values.append(round(average_mc_expected_value))
        standard_deviations_mc_expected_values.append(standard_deviation_mc_expected_value)
        coefficients_of_variation_mc_expected_values.append(coefficient_of_variation_mc_expected_value)  # Add coefficient of variation to the list

    df['CC_ST'] = list(gt_df['CC_ST'])
    df.insert(len(df.columns)-4, 'avg_soft_mc_pred', average_mc_expected_values)
    df.insert(len(df.columns)-4, 'rounded_avg_soft_mc_pred', rounded_average_mc_expected_values)
    df.insert(len(df.columns)-4, 'std_avg_soft_mc_pred', standard_deviations_mc_expected_values)
    df.insert(len(df.columns)-4, 'cv_avg_soft_mc_pred', coefficients_of_variation_mc_expected_values)  # Insert coefficient of variation column

    average_mc_probs = []
    for index, row in df.iterrows():
        class_0 = []
        class_1 = []
        if len(mc_probs) == 3:
            class_2 = []
        for col in mc_cols:
            mc_probs = ast.literal_eval(row[col])
            class_0.append(mc_probs[0])
            class_1.append(mc_probs[1])
            if len(mc_probs) == 3:
                class_2.append(mc_probs[2])
        if len(mc_probs) == 3:
            average_mc_probs.append([np.average(class_0), np.average(class_1), np.average(class_2)])
        else:
            average_mc_probs.append([np.average(class_0), np.average(class_1)])
    df.insert(len(df.columns)-4, 'avg_mc_probs_per_class', average_mc_probs)
    df.insert(len(df.columns)-4, 'argmax_mc_probs_per_class', np.argmax(np.array(average_mc_probs), axis=1))

    return df

def calculate_coeff_of_var(df, num_mc, num_classes):
    """
    Calculate the coefficients of variation for Monte Carlo predictions.

    Parameters:
    - df: DataFrame containing the Monte Carlo predictions in string format.
    - num_mc: Number of Monte Carlo samples.
    - num_classes: Number of classes in the classification task.

    Returns:
    - df: DataFrame with an additional column 'cov' containing the coefficients of variation for each image.
    """

    mc_predictions_cols = ['mc_epistemic_' + str(i) for i in range(num_mc)]
    mc_predictions = df[mc_predictions_cols].applymap(ast.literal_eval).values  # Convert strings to lists

    # Reshape mc_predictions to have three dimensions
    num_images = mc_predictions.shape[0]
    num_mc_samples = mc_predictions.shape[1]
    mc_predictions_reshaped = np.empty((num_images, num_mc_samples, num_classes), dtype=object)

    # Populate the reshaped array with the Monte Carlo predictions
    for i in range(num_images):
        for j in range(num_mc_samples):
            mc_pred_list = mc_predictions[i, j]
            for k in range(num_classes):
                mc_predictions_reshaped[i, j, k] = mc_pred_list[k]

    # Calculate the coefficients of variation
    coefficients_of_variation = []

    for i in range(num_images):
        mc_preds_for_image = mc_predictions_reshaped[i]  # Shape: (num_MC_samples, num_classes)
        evs = []
        evs2 = []

        # Calculate expected values for each Monte Carlo sample
        for mc_sample in range(num_mc_samples):
            expected_value = np.dot(mc_preds_for_image[mc_sample], np.arange(num_classes))
            evs.append(expected_value)
            evs2.append(expected_value + 1)
        
        # Compute standard deviation and mean of the expected values
        std_pred = np.std(evs)
        mean_pred = np.mean(evs)

        std_pred2 = np.std(evs2)
        mean_pred2 = np.mean(evs2)
        
        # Avoid division by zero by adding a small epsilon to the mean_pred if it's zero
        if mean_pred == 0:
            mean_pred += 1e-10
        
        # Calculate the coefficient of variation 
        coefficient_of_variation = (std_pred2 / mean_pred2) * 100
        coefficients_of_variation.append(coefficient_of_variation)

    # Convert the list of coefficients of variation to a numpy array
    coefficients_of_variation_array = np.array(coefficients_of_variation)

    # Add the coefficients of variation as a new column in the DataFrame
    df['cov'] = coefficients_of_variation_array

    return df

def convert_strings_to_list(list_of_strings):
    return [ast.literal_eval(string) for string in list_of_strings]

def create_cuq_df(cuq_type, predictions_df, gt_col, pred_probs_col, pred_class_col, img_id_col, alpha, pct_cal, print_or_not, num_mc, num_classes):
    """
    Create a DataFrame containing conformal uncertainty quantification (CUQ) information.

    Parameters:
    - cuq_type: Type of CUQ algorithm ('lac' or 'aps').
    - predictions_df: DataFrame containing predictions and other relevant data.
    - gt_col: Column name for ground truth values.
    - pred_probs_col: Column name for predicted probabilities.
    - pred_class_col: Column name for predicted classes (optional).
    - img_id_col: Column name for image IDs.
    - alpha: Significance level for prediction intervals.
    - pct_cal: Proportion of data to be used for calibration.
    - print_or_not: Boolean flag to print results.
    - num_mc: Number of Monte Carlo samples.
    - num_classes: Number of classes in the classification task.

    Returns:
    - cuq_df: DataFrame with CUQ results and additional metrics.
    """

    # Organizing our data
    gt_values = list(predictions_df[gt_col])
    pred_probs = convert_strings_to_list(list(predictions_df[pred_probs_col]))
    cov_values = list(calculate_coeff_of_var(predictions_df, num_mc, num_classes)['cov'])
    if pred_class_col is not None:
        pred_class = list(predictions_df[pred_class_col])
    else:
        print('We have no dedicated predicted class, so we are argmaxing over the given probability columns.')
        pred_class = [np.argmax(pred_prob) for pred_prob in pred_probs]
    img_ids = list(predictions_df[img_id_col])

    # Division into calibration and testing
    test_names, cal_names, \
    test_pred_classes, cal_pred_classes, \
    test_probs, cal_probs, \
    test_gt, cal_gt, \
    test_cov, cal_cov = train_test_split(img_ids, pred_class, np.array(pred_probs), np.array(gt_values).astype(int), cov_values, test_size=pct_cal, random_state=0)

    # Getting the calibration scores
    if cuq_type == 'lac':
        cal_scores = get_calibration_scores_lac(cal_gt, cal_probs)
    elif cuq_type == 'aps':
        cal_scores = get_calibration_scores_aps(cal_gt, cal_probs)
    else:
        print('Currently, our only two options for CUQ algorithms are lac and aps. Exiting')
        sys.exit()

    # Prediction sets
    if cuq_type == 'lac':
        try:
            prediction_sets, mean_width, covered, pct_coverage, qhat = cuq_pred_set_lac(cal_scores, alpha, test_probs, test_gt)
        except:
            prediction_sets, mean_width, qhat = cuq_pred_set_lac(cal_scores, alpha, test_probs, test_gt)
    elif cuq_type == 'aps':
        try:
            prediction_sets, mean_width, covered, pct_coverage, qhat = cuq_pred_set_aps(cal_scores, alpha, test_probs, test_gt)
        except:
            prediction_sets, mean_width, qhat = cuq_pred_set_aps(cal_scores, alpha, test_probs, test_gt)

    # Printing
    if print_or_not:
        print(f'Example prediction set: {prediction_sets[0]}')
        print(f'Mean width: {mean_width}')
        try:
            print(f'Percent/Empirical Coverage {pct_coverage}')
        except:
            print('No percent coverage')
        print(f'Associated q_hat: {qhat}')

    # Convert list of lists to a list of tuples to create a Pandas Series
    prediction_sets_series = pd.Series([tuple(item) for item in prediction_sets])
    prediction_set_length = [sum(item) for item in prediction_sets_series]

    # Create a dictionary with all the columns
    try:
        cuq_dict = {
            'image': test_names,
            'probs': list(test_probs),
            'pred_class': test_pred_classes,
            'pred_set': prediction_sets_series,
            'covered': covered,
            'pred_set_length': prediction_set_length,
            'gt': test_gt,
            'cov': test_cov
        }
    except:
        cuq_dict = {
            'image': test_names,
            'probs': list(test_probs),
            'pred_class': test_pred_classes,
            'pred_set': prediction_sets_series,
            'pred_set_length': prediction_set_length,
            'gt': test_gt,
            'cov': test_cov
        }

    # Create a DataFrame from the dictionary
    cuq_df = pd.DataFrame(cuq_dict)

    # Adding soft prediction
    cuq_df['soft_prediction'] = predictions_df[predictions_df['MASKED_IMG_ID'].isin(test_names)]['soft_prediction'].tolist()
    
    # Misclassifications
    cuq_df['correct_prediction'] = cuq_df['gt'] == cuq_df['pred_class']
    ## Calculate absolute difference between ground truth and predicted class
    diff = cuq_df['gt'] - cuq_df['pred_class']
    abs_diff = abs(cuq_df['gt'] - cuq_df['pred_class'])
    cuq_df['extreme_misclassification'] = abs_diff == (num_classes - 1)  # Extreme misclassification: difference == 2 for three-class, 1 for two-class
    if num_classes == 3:
        cuq_df['normal_misclassified_as_pcplus'] = (cuq_df['gt'] == 0) & (cuq_df['pred_class'] == 2)
        cuq_df['pcplus_misclassified_as_normal'] = (cuq_df['gt'] == 2) & (cuq_df['pred_class'] == 0)
        cuq_df['normal_misclassified_as_gz'] = (cuq_df['gt'] == 0) & (cuq_df['pred_class'] == 1)
        cuq_df['gz_misclassified_as_normal'] = (cuq_df['gt'] == 1) & (cuq_df['pred_class'] == 0)
        cuq_df['pcplus_misclassified_as_gz'] = (cuq_df['gt'] == 2) & (cuq_df['pred_class'] == 1)
        cuq_df['gz_misclassified_as_pcplus'] = (cuq_df['gt'] == 1) & (cuq_df['pred_class'] == 2)
    elif num_classes == 2:
        cuq_df['misclassified_as_opposite_class'] = (cuq_df['gt'] != cuq_df['pred_class'])
    cuq_df['cov'] = cuq_df['cov']

    return cuq_df

def create_mc_df(cuq_type, predictions_df, cuq_df_eval, label_col_name, alpha, num_mc_cols, num_classes, save_dir):
    """
    Create a DataFrame for Monte Carlo (MC) predictions and evaluate their performance.

    Parameters:
    - cuq_type: Type of CUQ algorithm ('lac' or 'aps').
    - predictions_df: DataFrame containing predictions and other relevant data.
    - cuq_df_eval: Evaluation DataFrame containing CUQ evaluation results.
    - label_col_name: Column name for ground truth labels.
    - alpha: Significance level for prediction intervals.
    - num_mc_cols: Number of Monte Carlo columns.
    - num_classes: Number of classes in the classification task.
    - save_dir: Directory to save the resulting DataFrame as a CSV file.

    Note, this function will calculate a conformal prediction for each MC run, and then vote for each class on whether that 
    class is included in the final conformal prediction set for the datum. 

    Returns:
    - None: Full Conformal Prediction dataframe ready for further analysis
    """

    mc_df = pd.DataFrame()
    for i in range(num_mc_cols):
        mc_col = 'mc_epistemic_' + str(i)
        cuq_df = create_cuq_df(cuq_type, predictions_df, label_col_name, mc_col, 'rounded_avg_soft_mc_pred', 'MASKED_IMG_ID', alpha, 0.2, False, num_mc_cols, num_classes)
        mc_df['pred_set_' + str(i)] = cuq_df['pred_set']

    mc_df.insert(0, 'image', cuq_df['image'])
    mc_df.insert(1, 'gt', cuq_df['gt'])
    mc_df.insert(2, 'pred_class_mc', cuq_df['pred_class'])
    mc_df.insert(3, 'pred_class_eval', cuq_df_eval['pred_class'])

    mc_df.insert(4, 'soft_mc_prediction', predictions_df[predictions_df['MASKED_IMG_ID'].isin(cuq_df['image'])]['soft_mc_prediction'].tolist())
    mc_df.insert(5, 'soft_eval_prediction', cuq_df_eval['soft_prediction'])

    mc_df = pd.merge(mc_df, predictions_df[['MASKED_IMG_ID', 'std_avg_soft_mc_pred']], left_on='image', right_on='MASKED_IMG_ID', how='inner')
    mc_df.pop('MASKED_IMG_ID')
    std_avg_soft_mc_pred = mc_df['std_avg_soft_mc_pred']
    mc_df.pop('std_avg_soft_mc_pred')
    mc_df.insert(6, 'std_avg_soft_mc_pred', std_avg_soft_mc_pred)

    voting_outcomes = []
    for index, row in mc_df.iterrows():  # Iterating over the rows
        class_vals = [[] for _ in range(num_classes)]
        for i in range(num_mc_cols):  # Iterating over the pred_set columns
            for j in range(num_classes):  # Iterating over the pred set indices
                val = row['pred_set_' + str(i)][j]
                class_vals[j].append(val)
        final_vote = [np.average(class_vals[k]) >= .5 for k in range(len(class_vals))]
        voting_outcomes.append(final_vote)

    mc_covered = np.array(voting_outcomes)[np.arange(len(mc_df)), list(mc_df['gt'])]
    mc_df.insert(7, 'pred_set_mc', voting_outcomes)
    mc_df.insert(8, 'pred_set_eval', cuq_df_eval['pred_set'])
    mc_df.insert(9, 'covered_by_pred_set_mc', mc_covered)
    mc_df.insert(10, 'covered_by_pred_set_eval', cuq_df_eval['covered'])
    mc_df.insert(11, 'pred_set_mc_length', [sum(pred_set) for pred_set in voting_outcomes])
    mc_df.insert(12, 'pred_set_eval_length', cuq_df_eval['pred_set_length'])

    # Misclassifications
    diff = mc_df['gt'] - mc_df['pred_class_mc']
    abs_diff = np.abs(diff)

    # Add new columns indicating extreme misclassifications for MC
    correct_prediction = mc_df['gt'] == mc_df['pred_class_mc']
    mc_df.insert(13, 'correct_prediction_mc', correct_prediction)
    extreme_misclassification = abs_diff == (num_classes - 1)  # Extreme misclassification: difference == 2 for three-class, 1 for two-class
    mc_df.insert(14, 'extreme_misclassification_mc', extreme_misclassification)

    if num_classes == 3:
        mc_df.insert(15, 'normal_misclassified_as_pcplus_mc', (mc_df['gt'] == 0) & (mc_df['pred_class_mc'] == 2))
        mc_df.insert(16, 'pcplus_misclassified_as_normal_mc', (mc_df['gt'] == 2) & (mc_df['pred_class_mc'] == 0))
        mc_df.insert(17, 'normal_misclassified_as_gz_mc', (mc_df['gt'] == 0) & (mc_df['pred_class_mc'] == 1))
        mc_df.insert(18, 'gz_misclassified_as_normal_mc', (mc_df['gt'] == 1) & (mc_df['pred_class_mc'] == 0))
        mc_df.insert(19, 'pcplus_misclassified_as_gz_mc', (mc_df['gt'] == 2) & (mc_df['pred_class_mc'] == 1))
        mc_df.insert(20, 'gz_misclassified_as_pcplus_mc', (mc_df['gt'] == 1) & (mc_df['pred_class_mc'] == 2))
    elif num_classes == 2:
        mc_df.insert(15, 'misclassified_as_opposite_class_mc', (mc_df['gt'] != mc_df['pred_class_mc']))

    # Add new columns indicating extreme misclassifications for eval
    mc_df.insert(21, 'correct_prediction_eval', cuq_df_eval['correct_prediction'])
    mc_df.insert(22, 'extreme_misclassification_eval', cuq_df_eval['extreme_misclassification'])

    if num_classes == 3:
        mc_df.insert(23, 'normal_misclassified_as_pcplus_eval', cuq_df_eval['normal_misclassified_as_pcplus'])
        mc_df.insert(24, 'pcplus_misclassified_as_normal_eval', cuq_df_eval['pcplus_misclassified_as_normal'])
        mc_df.insert(25, 'normal_misclassified_as_gz_eval', cuq_df_eval['normal_misclassified_as_gz'])
        mc_df.insert(26, 'gz_misclassified_as_normal_eval', cuq_df_eval['gz_misclassified_as_normal'])
        mc_df.insert(27, 'pcplus_misclassified_as_gz_eval', cuq_df_eval['pcplus_misclassified_as_gz'])
        mc_df.insert(28, 'gz_misclassified_as_pcplus_eval', cuq_df_eval['gz_misclassified_as_pcplus'])
    elif num_classes == 2:
        mc_df.insert(23, 'misclassified_as_opposite_class_eval', (cuq_df_eval['gt'] != cuq_df_eval['pred_class']))

    mc_df.insert(29, 'cov_eval', cuq_df_eval['cov'])
    mc_df.insert(30, 'cov_mc', cuq_df['cov'])

    if cuq_type == 'lac':
        mc_df.to_csv(os.path.join(save_dir, 'mc_lac_run_alpha' + str(alpha) + '.csv'))
    elif cuq_type == 'aps':
        mc_df.to_csv(os.path.join(save_dir, 'mc_aps_run_alpha' + str(alpha) + '.csv'))

    return mc_df

if __name__ == '__main__':
    # predictions_csv_path = '/sddata/projects/Cervical_Cancer_Projects/cervical_cancer_diagnosis/csvs/misc/model_36_test2_predictions_all_cols.csv'
    predictions_csv_path = '/sddata/projects/Cervical_Cancer_Projects/cervical_cancer_diagnosis/csvs/model_36_predictions/model_36_test2_predictions_all_cols.csv'
    # predictions_csv_path = '/sddata/projects/Cervical_Cancer_Projects/cervical_cancer_diagnosis/csvs/model_36_predictions/model_36_test2_binary.csv'
    cuq_type = 'lac'
    alpha = 0.1
    save_dir = '/sddata/projects/Conformal_Uncertainty_Quantification/MICCAI_UNSURE_Workshop/analysis/testing'
    num_mc_cols = 50
    num_classes = 3  # Change to 2 for two-class model
    label_col_name = 'CC_ST'

    predictions_df = pd.read_csv(predictions_csv_path)  # Load the predictions DataFrame
    cuq_df_eval = create_cuq_df(cuq_type, predictions_df, label_col_name, 'pred', 'predicted_class', 'MASKED_IMG_ID', alpha, 0.2, True, num_mc_cols, num_classes)
    mc_df = create_mc_df(cuq_type, predictions_df, cuq_df_eval, label_col_name, alpha, num_mc_cols, num_classes, save_dir)