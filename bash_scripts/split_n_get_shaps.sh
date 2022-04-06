# This script initializes the pipeline by setting up the experiment directories, splitting the deep learning embeddings and metadata by experiment, then it submits SLURM jobs to calculate the SHAP values for each deep learning embedding feature for predicting plate and disease condition.

embeddings=$1
metadata=$2

mkdir ../data/HRCE-1
mkdir ../data/HRCE-2
mkdir ../data/VERO-1
mkdir ../data/VERO-2

# Split data by experiment
source ../env/bin/activate
python3 ../python_scripts/split_by_experiment.py -e "$embeddings" -m "$metadata" -o ../data/ -n embeddings_and_metadata.csv

# Calculate plate shap values
sbatch get_plate_shaps.sh ../data/HRCE-1/embeddings_and_metadata.csv "HRCE-1"
sbatch get_plate_shaps.sh ../data/HRCE-2/embeddings_and_metadata.csv "HRCE-2"
sbatch get_plate_shaps.sh ../data/VERO-1/embeddings_and_metadata.csv "VERO-1"
sbatch get_plate_shaps.sh ../data/VERO-2/embeddings_and_metadata.csv "VERO-2"

# Calculate disease condition shap values
sbatch get_disease_condition_shaps.sh ../data/HRCE-1/embeddings_and_metadata.csv "HRCE-1"
sbatch get_disease_condition_shaps.sh ../data/HRCE-2/embeddings_and_metadata.csv "HRCE-2"
sbatch get_disease_condition_shaps.sh ../data/VERO-1/embeddings_and_metadata.csv "VERO-1"
sbatch get_disease_condition_shaps.sh ../data/VERO-2/embeddings_and_metadata.csv "VERO-2"