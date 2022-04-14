# This script calculates the cosine similarity between the deep learning embeddings of cells treated with SARS-CoV-2 and positive controls ("healthy" cells).

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import constants


def parse_args():
    """
    Defines arguments.
    
    Returns:
        args: some sort of argparse object that the arguments can be extracted from.
    """
    parser = argparse.ArgumentParser(description='Use a random grid search a random forest model to classify plate and then calculate the SHAP value for each feature.')
    parser.add_argument('-i', '--input', help='CSV of DL embeddings and metadata for a specific experiment.', type=str, required=True)
    parser.add_argument('-s', '--shap', help='CSV of disease condition SHAP values for a specific experiment.', type=str, required=False)
    parser.add_argument('-c', '--cutoffSHAP', help='SHAP value cutoff', type=float, default=0, required=False)
    parser.add_argument('-o', '--output', help='Directory to write output to. Default is current directory.', type=str, default='.', required=False)
    parser.add_argument('-f', '--outputFileName', help='Name of the ouput file.', type=str, default='cosine_similarity.csv', required=False)
    parser.add_argument('-e', '--expName', help='Experiment name.', type=str, default='EXPER_', required=False)
    parser.add_argument('-n', '--normalized', help='Flag to indicate input is normalized, shap files do not exist, and no feature dropping is necessary.', action='store_true', required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Main."""
    
    # Parse arguments.
    args = parse_args()
    out_dir = args.output.rstrip('/')
    file_name = args.outputFileName
    cutoff = args.cutoffSHAP
    e = args.expName
    out_file_name = e + '_' + file_name
    
    # Load data and preprocess data.
    joined_df = pd.read_csv(args.input)

    # fill disease_condition=="" with 'null'
    joined_df[constants.DISEASE_CONDITION] = joined_df[constants.DISEASE_CONDITION].fillna('null')

    # Combinde experiment, treatment and treatment concentration information
    joined_df[constants.TREATMENT_CONC] = joined_df[constants.TREATMENT_CONC].astype(str)
    joined_df.fillna('',inplace=True)
    joined_df[constants.EXPERIMENT_TRT_CONC] = joined_df[constants.EXPERIMENT].astype(str) + '/' + joined_df[constants.TREATMENT] + '/' + joined_df[constants.TREATMENT_CONC].astype(str)
    
    if not args.normalized:
        shap = pd.read_csv(args.shap)  # read in corresponding shap values

        # drop SHAP values less than a prescribed cutoff
        features_to_drop = list(shap[shap[constants.SHAP_VALUE] < cutoff].feature)
        joined_df_dot = joined_df.copy(deep=True).drop(features_to_drop,axis=1)
    
    else:
        joined_df_dot = joined_df

    # Get relevant columns
    joined_df_dot.drop([constants.PLATE, constants.SMILES, constants.TREATMENT, constants.SITE, constants.WELL, constants.EXPERIMENT, constants.CELL_TYPE, constants.WELL_ID, constants.SITE_ID, constants.TREATMENT_CONC], axis=1, inplace=True)
    
    # Get active disease data
    active_sars_cov_only = joined_df_dot[joined_df_dot[constants.DISEASE_CONDITION] == 'Active SARS-CoV-2']
    active_sars_cov_only[constants.EXPERIMENT_TRT_CONC_DC] = active_sars_cov_only[constants.EXPERIMENT_TRT_CONC] + ' ' + active_sars_cov_only[constants.DISEASE_CONDITION]
    active_sars_cov_only_groupby_mean = active_sars_cov_only.groupby(constants.EXPERIMENT_TRT_CONC_DC).mean()
    
    # get control data
    controls_only = joined_df_dot[joined_df_dot[constants.DISEASE_CONDITION] != 'Active SARS-CoV-2']
    controls_only[constants.EXPERIMENT_TRT_CONC_DC] = controls_only[constants.EXPERIMENT_TRT_CONC] + ' ' + controls_only[constants.DISEASE_CONDITION]
    controls_only_groupby_mean = controls_only.groupby(constants.EXPERIMENT_TRT_CONC_DC).mean()
    
    # Get cosine similarity
    cosine_similarity_matrix = cosine_similarity(active_sars_cov_only_groupby_mean , controls_only_groupby_mean)
    cosine_sim_df = pd.DataFrame(data=cosine_similarity_matrix,index = active_sars_cov_only_groupby_mean.index.to_list(),columns = controls_only_groupby_mean.index.to_list())
    
    # Write to csv
    cosine_sim_df.to_csv(f'{out_dir}/{out_file_name}', index=True)
