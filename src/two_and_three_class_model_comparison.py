import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
## Stats Testing
from scipy import stats

def independent_ttest(data1, data2, alpha=0.05):
    """
    Perform an independent samples t-test.

    Parameters:
    - data1: list or array-like, containing the data for the first group
    - data2: list or array-like, containing the data for the second group
    - alpha: significance level for the confidence interval

    Returns:
    - t_statistic: The calculated t-statistic
    - p_value: The two-tailed p-value
    - ci_lower1: Lower bound of the confidence interval for the mean of data1
    - ci_upper1: Upper bound of the confidence interval for the mean of data1
    - ci_lower2: Lower bound of the confidence interval for the mean of data2
    - ci_upper2: Upper bound of the confidence interval for the mean of data2
    """
    t_statistic, p_value = stats.ttest_ind(data1, data2)
    
    # Degrees of freedom
    dof = len(data1) + len(data2) - 2
    
    # Standard error of the mean for each group
    se1 = np.std(data1, ddof=1) / np.sqrt(len(data1))
    se2 = np.std(data2, ddof=1) / np.sqrt(len(data2))
    
    # Critical value
    t_critical = stats.t.ppf(1 - alpha / 2, dof)
    
    # Confidence intervals
    mean_diff = np.mean(data1) - np.mean(data2)
    ci_lower1 = np.mean(data1) - t_critical * se1
    ci_upper1 = np.mean(data1) + t_critical * se1
    ci_lower2 = np.mean(data2) - t_critical * se2
    ci_upper2 = np.mean(data2) + t_critical * se2

    return t_statistic, p_value, data1.mean(), data2.mean(), ci_lower1, ci_upper1, ci_lower2, ci_upper2

def compare_conformal_prediction_lengths(cuq_df_2c, cuq_df_3c, mc_or_eval, save_path, save_name):
    """
    Compare the conformal prediction lengths between the two-class and three-class models.

    Parameters:
    - cuq_df_2c: DataFrame containing the CUQ results for the two-class model.
    - cuq_df_3c: DataFrame containing the CUQ results for the three-class model.
    - mc_or_eval: String of 'mc' or 'eval' to determine what we are comparing.
    - save_path: Directory to save the resulting files.
    - save_name: Base name for the saved files.

    Returns:
    - None: The function saves the resulting analysis as CSV files.
    """

    if mc_or_eval == 'eval':
        pred_set_length_col = 'pred_set_eval_length'
    elif mc_or_eval == 'mc':
        pred_set_length_col = 'pred_set_mc_length'
    else:
        raise ValueError('Please choose either "mc" or "eval" to determine the comparison.')

    # Create mappings for comparison
    cuq_df_2c['gt_3c'] = cuq_df_3c['gt']  # Mapping the three-class ground truth to the two-class dataframe

    comparisons = [
        ("GZ in 3C vs Overall 2C (Including GZ)", cuq_df_2c[cuq_df_2c['gt_3c'] == 1][pred_set_length_col], cuq_df_2c[pred_set_length_col]),
        ("GZ in 3C vs Not GZ in 2C", cuq_df_2c[cuq_df_2c['gt_3c'] == 1][pred_set_length_col], cuq_df_2c[cuq_df_2c['gt_3c'] != 1][pred_set_length_col]),
        ("GZ in 3C vs Normal in 2C (Including GZ)", cuq_df_2c[cuq_df_2c['gt_3c'] == 1][pred_set_length_col], cuq_df_2c[cuq_df_2c['gt'] == 0][pred_set_length_col]),
        ("GZ in 3C vs PCPlus in 2C (Including GZ)", cuq_df_2c[cuq_df_2c['gt_3c'] == 1][pred_set_length_col], cuq_df_2c[cuq_df_2c['gt'] == 1][pred_set_length_col]),
        ("GZ in 3C vs Normal in 2C (Excluding GZ)", cuq_df_2c[cuq_df_2c['gt_3c'] == 1][pred_set_length_col], cuq_df_2c[(cuq_df_2c['gt'] == 0) & (cuq_df_2c['gt_3c'] != 1)][pred_set_length_col]),
        ("GZ in 3C vs PCPlus in 2C (Excluding GZ)", cuq_df_2c[cuq_df_2c['gt_3c'] == 1][pred_set_length_col], cuq_df_2c[(cuq_df_2c['gt'] == 1) & (cuq_df_2c['gt_3c'] != 1)][pred_set_length_col]),
        ("Normal in 2C (Including GZ) vs PCPlus in 2C (Including GZ)", cuq_df_2c[cuq_df_2c['gt'] == 0][pred_set_length_col], cuq_df_2c[cuq_df_2c['gt'] == 1][pred_set_length_col]),
        ("Normal in 2C (Excluding GZ) vs PCPlus in 2C (Excluding GZ)", cuq_df_2c[(cuq_df_2c['gt'] == 0) & (cuq_df_2c['gt_3c'] != 1)][pred_set_length_col], cuq_df_2c[(cuq_df_2c['gt'] == 1) & (cuq_df_2c['gt_3c'] != 1)][pred_set_length_col])
    ]

    names, data1_means, data2_means, t_statistics, p_values = [], [], [], [], []
    ci_lowers1, ci_uppers1, ci_lowers2, ci_uppers2 = [], [], [], []

    for name, data1, data2 in comparisons:
        t_statistic, p_value, data1_mean, data2_mean, ci_lower1, ci_upper1, ci_lower2, ci_upper2 = independent_ttest(data1, data2)
        names.append(name)
        t_statistics.append(t_statistic)
        p_values.append(p_value)
        data1_means.append(data1_mean)
        data2_means.append(data2_mean)
        ci_lowers1.append(ci_lower1)
        ci_uppers1.append(ci_upper1)
        ci_lowers2.append(ci_lower2)
        ci_uppers2.append(ci_upper2)

    comparison_t_test_dict = {
        'Type': names,
        'Data1 Mean': data1_means,
        'Data2 Mean': data2_means,
        'ci_lower1': ci_lowers1,
        'ci_upper1': ci_uppers1,
        'ci_lower2': ci_lowers2,
        'ci_upper2': ci_uppers2,
        't_statistic': t_statistics,
        'p_value': p_values
    }

    comparison_df = pd.DataFrame(comparison_t_test_dict)
    comparison_df.to_csv(os.path.join(save_path, 'comparison_' + save_name))

if __name__ == '__main__':

    cuq_df_3c_path = '/sddata/projects/Conformal_Uncertainty_Quantification/MICCAI_UNSURE_Workshop_Anon_Git/analysis/3/lac/alpha_010/mc_lac_run_alpha0.1.csv'
    cuq_df_3c = pd.read_csv(cuq_df_3c_path)
    
    cuq_df_2c_path = '/sddata/projects/Conformal_Uncertainty_Quantification/MICCAI_UNSURE_Workshop_Anon_Git/analysis/2/lac/alpha_010/mc_lac_run_alpha0.1.csv'
    cuq_df_2c = pd.read_csv(cuq_df_2c_path)

    save_path = '/sddata/projects/Conformal_Uncertainty_Quantification/MICCAI_UNSURE_Workshop_Anon_Git/analysis'

    compare_conformal_prediction_lengths(cuq_df_2c, cuq_df_3c, 'mc', save_path, 'conformal_comparison_lac.csv')
