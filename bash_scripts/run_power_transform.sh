#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH -J "pwr trnsf" # job name

# This script power transforms the embeddings to remove batch effects.

# Arguments: (1) experiment
source env/bin/activate
python3 python_scripts/power_transform.py -i data/"$1"/embeddings_and_metadata.csv -o data/"$1"