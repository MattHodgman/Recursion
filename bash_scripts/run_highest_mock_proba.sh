#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH -J "mock proba" # job name

# This script calculates the predicted probability that tests of compounds are missclassified as positive controls ("healthy" cells).

# $1=experiment_data $2=experiment $3=final_shap_cutoff
source env/bin/activate

if [ "$2" == "n" ]
    then
        python3 python_scripts/highest_mock_proba.py -i $1 -n -o data/$2
    else
        python3 python_scripts/highest_mock_proba.py -i $1 -s data/$2/shap_values_disease_condition.csv -o data/$2 -f $3
fi