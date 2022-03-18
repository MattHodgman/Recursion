import pandas as pd
import numpy as np
import argparse
from imblearn.ensemble import BalancedRandomForestClassifier


# Constants.
_SHAP_VALUE = 'shap_value'
_FEATURE_PREFIX = 'feature_'


def parse_args():
    """
    Defines arguments.
    
    Returns:
        args: some sort of argparse object that the arguments can be extracted from.
    """
    parser = argparse.ArgumentParser(description='For a given experiment, obtain the treatment and concentration with the highest probability of being predicted as MOCK')
    parser.add_argument('-i', '--input', help='CSV of DL embeddings and metadata for a specific experiment.', type=str, required=True)
    parser.add_argument('-s', '--diseaseConditionShaps', help='A CSV with a column \'feature\' that contains feature IDs and a column \'shap_value\' that contains the Shapley value of that feature when predicting disease condition. ', type=str, required=True)
    parser.add_argument('-f', '--finalShapCutoff', help='The final Shapley value cutoff.', type=float, default=0.0023, required=False)
    parser.add_argument('-o', '--output', help='Directory to write output to. Default is current directory.', type=str, default='.', required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Main."""

    # Parse arguments.
    args = parse_args()
    out_dir = args.output.rstrip('/')
    final_shap_cutoff = args.finalShapCutoff

    # Load data.
    df = pd.read_csv(args.input)
    shap = pd.read_csv(args.diseaseConditionShaps)

    # Preprocess data.
    shap = shap.sort_values(by=[_SHAP_VALUE], ascending=False)
    features_to_drop = list(shap[shap[_SHAP_VALUE]<final_shap_cutoff].feature)
    joined_df_dot = df.copy(deep=True).drop(features_to_drop,axis=1)
    df['disease_condition'] = df['disease_condition'].fillna('null')
    joined_df_dot['disease_condition'] = joined_df_dot['disease_condition'].fillna('null')
    df['treatment_conc'] = df['treatment_conc'].astype(str)
    df = df[~df['treatment'].isna()]
    df = df.fillna('')
    df['treatment_plus_conc'] = df['treatment'] + " " + df['treatment_conc'].astype(str)

    # Split into control and treatment.
    feature_cols = [col for col in joined_df_dot.columns if _FEATURE_PREFIX in col]
    joined_df_dot_controls = joined_df_dot[joined_df_dot['treatment'].isna()]
    X_train = joined_df_dot_controls[feature_cols]
    y_train = joined_df_dot_controls['disease_condition']

    joined_df_dot_treatments = joined_df_dot[~joined_df_dot['treatment'].isna()]
    X_test = joined_df_dot_treatments[feature_cols]
    y_test = joined_df_dot_treatments['disease_condition']
    
    # Balanced Random Forest
    brc = BalancedRandomForestClassifier()
    brc.fit(X_train, y_train)
    predicted_prob = brc.predict_proba(X_test)
    df['Predicted Mock'] = list(predicted_prob[:,1])
    
    # Of the individuals experiments that are assigned as Active SARS-CoV-2, which ones has the highest average probability to be predicted as mock
    mock_prob_given_active = df[df['disease_condition']=='Active SARS-CoV-2'].groupby('treatment_plus_conc')['Predicted Mock'].mean()
    mock_prob_given_active.sort_values(ascending=False).head(20).to_csv('{}/mock_proba_{}_{}_cutoff_brf.csv'.format(out_dir,final_shap_cutoff,out_dir))