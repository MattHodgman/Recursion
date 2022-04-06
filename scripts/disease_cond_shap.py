import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
import shap
import constants


def parse_args():
    """
    Defines arguments.
    
    Returns:
        args: some sort of argparse object that the arguments can be extracted from.
    """
    parser = argparse.ArgumentParser(description='Use a balanced random forest model to classify disease condition and then calculate the Shapley score for each feature.')
    parser.add_argument('-i', '--input', help='CSV of DL embeddings and metadata for a specific experiment.', type=str, required=True)
    parser.add_argument('-o', '--output', help='Directory to write output to. Default is current directory.', type=str, default='.', required=False)
    parser.add_argument('-n', '--outputFileName', help='Name of the ouput file.', type=str, default='shap_values_disease_condition.csv', required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Main."""

    # Parse arguments.
    args = parse_args()
    out_dir = args.output.rstrip('/')
    out_file_name = args.outputFileName

    # Load data and preprocess data.
    df = pd.read_csv(args.input)

    feature_cols = [col for col in df.columns if constants.FEATURE_PREFIX in col]
    df[constants.DISEASE_CONDITION] = df[constants.DISEASE_CONDITION].fillna(constants.NULL)

    # Split data to train on the controls and test on the experiments with drugs
    df_controls = df[df[constants.TREATMENT].isna()]
    X_train = df_controls[feature_cols]
    y_train = df_controls[constants.DISEASE_CONDITION]

    df_drugs = df[~df[constants.TREATMENT].isna()]
    X_test = df_drugs[feature_cols]
    y_test = df_drugs[constants.DISEASE_CONDITION]

    del df
    del df_controls
    del df_drugs

    # Run model to classify disease condition.
    brc = BalancedRandomForestClassifier()
    brc.fit(X_train, y_train)
    y_pred_brc = brc.predict(X_test)

    # Plot confusion matrix.
    plot_confusion_matrix(brc, X_test, y_test)
    plt.savefig(f'{out_dir}/confusion_matrix.png')

    # Calculate Shapley values.
    explainer = shap.TreeExplainer(brc)
    shap_values = explainer.shap_values(X_test)

    # Write feature Shapley values to CSV.
    mean_feature_shap_values = np.abs(shap_values).mean(0).mean(0)
    df_shap = pd.DataFrame(mean_feature_shap_values, columns=[constants.SHAP_VALUE])
    df_shap[constants.FEATURE] = constants.FEATURE_PREFIX + df_shap.index.astype(str)
    df_shap = df_shap[[constants.FEATURE, constants.SHAP_VALUE]]
    df_shap = df_shap.sort_values(by=constants.SHAP_VALUE, ascending=False)
    df_shap.to_csv(f'{out_dir}/{out_file_name}', index=False)