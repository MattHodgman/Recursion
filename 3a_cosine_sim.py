import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

_FEATURE = 'feature_'
_DISEASE_CONDITION = 'disease_condition'
_TREATMENT_CONC = 'treatment_conc'
_TREATMENT = 'treatment'
_EXPERIMENT = 'experiment'
_EXPERIMENT_TRT_CONC = 'expt_treatment_plus_conc'
_EXPERIMENT_TRT_CONC_DC = 'expt_treatment_plus_conc_dc'
_PLATE = 'plate'
_SMILES = 'SMILES'
_WELL = 'well'
_WELL_ID = 'well_id'
_SITE = 'site'
_SITE_ID = 'site_id'
_CELL_TYPE = 'cell_type'
_SHAP_VALUE = 'shap_value'


def parse_args():
    """
    Defines arguments.
    
    Returns:
        args: some sort of argparse object that the arguments can be extracted from.
    """
    parser = argparse.ArgumentParser(description='Use a random grid search a random forest model to classify plate and then calculate the SHAP value for each feature.')
    parser.add_argument('-i', '--input', help='CSV of DL embeddings and metadata for a specific experiment.', type=str, required=True)
    parser.add_argument('-s', '--shap', help='CSV of SHAP values for a specific experiment.', type=str, required=True)
    parser.add_argument('-c', '--cutoffSHAP', help='SHAP value cutoff', type=str, required=True)
    parser.add_argument('-o', '--output', help='Directory to write output to. Default is current directory.', type=str, default='.', required=False)
    parser.add_argument('-n', '--outputFileName', help='Name of the ouput file.', type=str, default='cosine_similarity.csv', required=False)
    parser.add_argument('-e', '--expName', help='Experiment name.', type=str, default='EXPER_', required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Main."""
    
    # Parse arguments.
    args = parse_args()
    out_dir = args.output.rstrip('/')
    file_name = args.outputFileName
    cutoff = float(args.cutoffSHAP)
    e = args.expName
    out_file_name = e + '_' + file_name
    
    # Load data and preprocess data.
    joined_df = pd.read_csv(args.input)
    # fill disease_condition=="" with 'null'
    joined_df[_DISEASE_CONDITION] = joined_df[_DISEASE_CONDITION].fillna('null')
    # Combinde experiment, treatment and treatment concentration information
    joined_df[_TREATMENT_CONC] = joined_df[_TREATMENT_CONC].astype(str)
    joined_df.fillna('',inplace=True)
    joined_df[_EXPERIMENT_TRT_CONC] = joined_df[_EXPERIMENT].astype(str) + '/' + joined_df[_TREATMENT] + '/' + joined_df[_TREATMENT_CONC].astype(str)
    
    # read in corresponding shap values
    shap = pd.read_csv(args.shap)
    # drop SHAP values less than a prescribed cutoff
    features_to_drop = list(shap[shap[_SHAP_VALUE] < cutoff].feature)
    joined_df_dot = joined_df.copy(deep=True).drop(features_to_drop,axis=1)
    # Get relevant columns
    joined_df_dot.drop([_PLATE, _SMILES, _TREATMENT, _SITE, _WELL, _EXPERIMENT, _CELL_TYPE, _WELL_ID, _SITE_ID, _TREATMENT_CONC], axis=1, inplace=True)
    # Get active disease data
    active_sars_cov_only = joined_df_dot[joined_df_dot[_DISEASE_CONDITION] == 'Active SARS-CoV-2']
    active_sars_cov_only[_EXPERIMENT_TRT_CONC_DC] = active_sars_cov_only[_EXPERIMENT_TRT_CONC] + ' ' + active_sars_cov_only[_DISEASE_CONDITION]
    active_sars_cov_only_groupby_mean = active_sars_cov_only.groupby(_EXPERIMENT_TRT_CONC_DC).mean()
    # get control data
    controls_only = joined_df_dot[joined_df_dot[_DISEASE_CONDITION] != 'Active SARS-CoV-2']
    controls_only[_EXPERIMENT_TRT_CONC_DC] = controls_only[_EXPERIMENT_TRT_CONC] + ' ' + controls_only[_DISEASE_CONDITION]
    controls_only_groupby_mean = controls_only.groupby(_EXPERIMENT_TRT_CONC_DC).mean()
    # Get cosing similarity
    cosine_similarity_matrix = cosine_similarity(active_sars_cov_only_groupby_mean , controls_only_groupby_mean)
    cosine_sim_df = pd.DataFrame(data=cosine_similarity_matrix,index = active_sars_cov_only_groupby_mean.index.to_list(),columns = controls_only_groupby_mean.index.to_list())
    # Write to csv
    cosine_sim_df.to_csv(f'{out_dir}/{out_file_name}', index=True)