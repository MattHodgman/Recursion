import pandas as pd
import numpy as np
import argparse
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt


# Constants. NOTE: I was too lazy to make a constant for each scoring metric.
_FEATURE_PREFIX = 'feature_'
_SHAP_VALUE = 'shap_value'
_SITE_ID = 'site_id'
_DISEASE_CONDITION = 'disease_condition'
_NULL = 'null'
_TREATMENT = 'treatment'


def parse_args():
    """
    Defines arguments.
    
    Returns:
        args: some sort of argparse object that the arguments can be extracted from.
    """
    parser = argparse.ArgumentParser(description='Perform a sensitivity analysis by incremently dropping features according to their Shapley score for predicting disease condition.')
    parser.add_argument('-i', '--input', help='CSV of DL embeddings and metadata for a specific experiment.', type=str, required=True)
    parser.add_argument('-s', '--diseaseConditionShaps', help='A CSV with a column \'feature\' that contains feature IDs and a column \'shap_value\' that contains the Shapley value of that feature when predicting disease condition. ', type=str, required=True)
    parser.add_argument('-f', '--finalShapCutoff', help='The final Shapley value cutoff.', type=float, default=0.0023, required=False)
    parser.add_argument('-c', '--CutoffStep', help='The Shapley value cutoff step size.', type=float, default=0.0002, required=False)
    parser.add_argument('-o', '--output', help='Directory to write output to. Default is current directory.', type=str, default='.', required=False)
    parser.add_argument('-n', '--outputFileName', help='Name of the ouput file.', type=str, default='sensitivity_disease_condition_shap_cutoff', required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Main."""

    # Parse arguments.
    args = parse_args()
    out_dir = args.output.rstrip('/')
    out_file_name = args.outputFileName
    final_shap_cutoff = args.finalShapCutoff
    cutoff_step = args.CutoffStep

    # Load data.
    df = pd.read_csv(args.input)
    shaps =  pd.read_csv(args.diseaseConditionShaps)

    # Preprocess data.
    shaps = shaps.sort_values(by=[_SHAP_VALUE], ascending=False)
    feature_cols = [col for col in df.columns if _FEATURE_PREFIX in col]
    df[_DISEASE_CONDITION] = df[_DISEASE_CONDITION].fillna(_NULL)

    # Split data to train on the controls and test on the experiments with drugs
    df_controls = df[df[_TREATMENT].isna()]
    X = df_controls[feature_cols]
    y = df_controls[_DISEASE_CONDITION]

    # Drop features based on Shapley score for predicting disease condition
    shap_cutoffs = np.arange(0, shaps[_SHAP_VALUE].max(), cutoff_step).tolist()
    sensitivity_df = pd.DataFrame(columns = ['Macro Precision','Macro Recall','Macro F1'], index=shap_cutoffs)

    n_features_at_cutoff = len(list(shaps[shaps[_SHAP_VALUE] > final_shap_cutoff].feature))
    n_features_list = []

    for i in shap_cutoffs:
        if i > 0 :
            features_to_drop = list(shaps[shaps[_SHAP_VALUE] < i].feature)
            X_dot = X.copy(deep=True).drop(features_to_drop, axis=1)
        elif i == 0:
            X_dot = X
            
        n_features = len(X_dot.columns)
        n_features_list.append(n_features)

        brc = BalancedRandomForestClassifier()
        cv_results = cross_validate(brc, X_dot, y, cv=5, scoring=('precision_macro', 'recall_macro', 'f1_macro'))
        
        sensitivity_df.loc[i,'Macro Precision'] = np.mean(cv_results['test_precision_macro'])
        sensitivity_df.loc[i,'Macro F1'] = np.mean(cv_results['test_f1_macro'])
        sensitivity_df.loc[i,'Macro Recall'] = np.mean(cv_results['test_recall_macro'])

    sensitivity_df['shap_cutoff'] = sensitivity_df.index
    sensitivity_df['feature_count'] = n_features_list
    sensitivity_df.to_csv(f'{out_dir}/{out_file_name}.csv')

    # Plot.
    plt.plot('shap_cutoff', 'Macro Precision', data=sensitivity_df, color='blue')
    plt.plot('shap_cutoff', 'Macro Recall', data=sensitivity_df, color='orange')
    plt.plot('shap_cutoff', 'Macro F1', data=sensitivity_df, color='grey')
    plt.axvline(x=final_shap_cutoff, color='black', label=final_shap_cutoff, dashes=(3,3))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'{out_dir} Sensitivity Test')
    plt.xlabel('shap cutoff')
    plt.ylabel('score')
    plt.savefig(f'{out_dir}/{out_file_name}_shaps.png', bbox_inches='tight')

    plt.clf()
    plt.plot('feature_count', 'Macro Precision', data=sensitivity_df, color='blue')
    plt.plot('feature_count', 'Macro Recall', data=sensitivity_df, color='orange')
    plt.plot('feature_count', 'Macro F1', data=sensitivity_df, color='grey')
    plt.axvline(x=n_features_at_cutoff, color='black', label=n_features_at_cutoff, dashes=(3,3))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'{out_dir} Sensitivity Test')
    plt.xlabel('feature count')
    plt.ylabel('score')
    plt.savefig(f'{out_dir}/{out_file_name}_n_features.png', bbox_inches='tight')