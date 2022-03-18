#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=30G
#SBATCH -J "dc shaps" # job name

data=$1 # the combined embeddings and metadata file for an experiment
experiment=$2 # the name of the experiment

# compute shap values for predicting disease condition.
source env/bin/activate
python3 1b_disease_cond_shap.py -i "$data" -o "$experiment" -n shap_values_disease_condition.csv