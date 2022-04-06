#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH -J "sensitivity test" # job name

# $1=experiment_data $2=experiment/shap_values_disease_condition.csv $3=output_dir $4=final_shap_cutoff
source ../env/bin/activate
python3 ../python_scripts/disease_condition_sensitivity_analysis.py -i $1 -s $2 -o $3 -f $4