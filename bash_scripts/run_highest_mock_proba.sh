#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH -J "mock proba" # job name

# $1=experiment_data $2=experiment $3=final_shap_cutoff
source ../env/bin/activate
python3 ../python_scripts/highest_mock_proba.py -i $1 -s ../data/$2/shap_values_disease_condition.csv -o ../data/$2 -f $3