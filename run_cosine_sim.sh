#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH -J "cosine" # job name

# Arguments (1) experiment (2) shap cutoff
source env/bin/activate
python3 3a_cosine_sim.py -i "$1"/embeddings_and_metadata.csv -s "$1"/shap_values_disease_condition.csv -c $2 -e "$1" -o $1