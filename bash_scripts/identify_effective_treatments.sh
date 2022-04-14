# This script uses the experiment-specific deep learning embeddings and disease condition SHAP cutoffs to rank compounds on their predicted effectiveness against SARS-CoV-2
# Arguments: the disease condition shap value cutoff for each experiment, or just 'n' when only normalizing.

HRCE1=$1
HRCE2=$2
VERO1=$3
VERO2=$4

# Handle normalization
if [ "$1" == "n" ]
    then
        HRCE1="n"
        HRCE2="n"
        VERO1="n"
        VERO2="n"
fi

# Compute cosine similarity of treatment embeddings to controls.
sbatch bash_scripts/run_cosine_sim.sh HRCE-1 $HRCE1
sbatch bash_scripts/run_cosine_sim.sh HRCE-2 $HRCE2
sbatch bash_scripts/run_cosine_sim.sh VERO-1 $VERO1
sbatch bash_scripts/run_cosine_sim.sh VERO-2 $VERO2

# Compute the probability of a treatment being classified as a control.
sbatch bash_scripts/run_highest_mock_proba.sh data/HRCE-1/embeddings_and_metadata.csv HRCE-1/ $HRCE1
sbatch bash_scripts/run_highest_mock_proba.sh data/HRCE-2/embeddings_and_metadata.csv HRCE-2/ $HRCE2
sbatch bash_scripts/run_highest_mock_proba.sh data/VERO-1/embeddings_and_metadata.csv VERO-1/ $VERO1
sbatch bash_scripts/run_highest_mock_proba.sh data/VERO-2/embeddings_and_metadata.csv VERO-2/ $VERO2

