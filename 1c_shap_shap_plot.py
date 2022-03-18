import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Constants
_FEATURE = 'feature'
_SHAP_VALUE = 'shap_value'

def parse_args():
    """
    Defines arguments.
    
    Returns:
        args: some sort of argparse object that the arguments can be extracted from.
    """
    parser = argparse.ArgumentParser(description='Create a scatter plot of features using their Shapley scores for predicting plate and disease condition.')
    parser.add_argument('-p', '--plateShaps', help='A CSV with a column \'feature\' that contains feature IDs and a column \'shap_value\' that contains the Shapley value of that feature when predicting plate.', type=str, required=True)
    parser.add_argument('-d', '--diseaseConditionShaps', help='A CSV with a column \'feature\' that contains feature IDs and a column \'shap_value\' that contains the Shapley value of that feature when predicting disease condition.', type=str, required=True)
    parser.add_argument('-o', '--output', help='Directory to write output to. Default is current directory.', type=str, default='.', required=False)
    parser.add_argument('-n', '--outputFileName', help='Name of the ouput file.', type=str, default='shap_v_shap.png', required=False)
    parser.add_argument('-e', '--experiment', help='Experiment ID (HRCE-1, etc.)', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """Main."""

    # Parse arguments.
    args = parse_args()
    out_dir = args.output.rstrip('/')
    out_file_name = args.outputFileName

    # Load Shapley values.
    plate_shaps = pd.read_csv(args.plateShaps)
    disease_condition_shaps = pd.read_csv(args.diseaseConditionShaps)

    # Remove 'feature_' from feature names, only retain integer ID.
    plate_shaps[_FEATURE] = plate_shaps[_FEATURE].str.replace(r'\D', '').astype(int)
    disease_condition_shaps[_FEATURE] = disease_condition_shaps[_FEATURE].str.replace(r'\D', '').astype(int)

    # Sort features by their ID.
    plate_shaps.sort_values(by=_FEATURE, ascending=True, inplace=True)
    disease_condition_shaps.sort_values(by=_FEATURE, ascending=True,inplace=True)

    # Plot features according to their Shapley values.
    plt.figure(figsize=(10,6))
    plt.scatter(x=plate_shaps[_SHAP_VALUE], y=disease_condition_shaps[_SHAP_VALUE])
    plt.xlabel('Plate Shap')
    plt.ylabel('Disease Condition Shap')
    plt.title(f'{args.experiment} Feature Importance in Predicting Plate vs. Disease Condition')
    plt.savefig(f'{out_dir}/{out_file_name}')