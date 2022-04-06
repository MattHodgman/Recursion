#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH -J "cosine" # job name

# This script calculates the cosine similarity between the deep learning embeddings of cells treated with SARS-CoV-2 and positive controls ("healthy" cells).

# Arguments (1) experiment (2) shap cutoff
source ../env/bin/activate
python3 ../python_scripts/cosine_sim.py -i ../data/"$1"/embeddings_and_metadata.csv -s ../data/"$1"/shap_values_disease_condition.csv -c $2 -e "$1" -o ../data/$1