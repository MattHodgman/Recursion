#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=30G
#SBATCH -J "plate shaps" # job name

data=$1 # the combined embeddings and metadata file for an experiment
experiment=$2 # the name of the experiment

# compute shap values for predicting plate.
source ../env/bin/activate
python3 ../python_scripts/plate_shap.py -i "$data" -o ../data/"$experiment" -n shap_values_plate.csv