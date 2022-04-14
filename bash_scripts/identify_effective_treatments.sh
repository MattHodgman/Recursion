# This script uses the experiment-specific deep learning embeddings and disease condition SHAP cutoffs to rank compounds on their predicted effectiveness against SARS-CoV-2
# Arguments: the disease condition shap value cutoff for each experiment, or just 'n' when only normalizing.

# Handle normalization
if [ "$1" == "n" ]
    then
        1="n"
        2="n"
        3="n"
        4="n"
fi

# Compute cosine similarity of treatment embeddings to controls.
sbatch bash_scripts/run_cosine_sim.sh HRCE-1 $1
sbatch bash_scripts/run_cosine_sim.sh HRCE-2 $2
sbatch bash_scripts/run_cosine_sim.sh VERO-1 $3
sbatch bash_scripts/run_cosine_sim.sh VERO-2 $4

# Compute the probability of a treatment being classified as a control.
sbatch bash_scripts/run_highest_mock_proba.sh data/HRCE-1/embeddings_and_metadata.csv HRCE-1/ $1
sbatch bash_scripts/run_highest_mock_proba.sh data/HRCE-2/embeddings_and_metadata.csv HRCE-2/ $2
sbatch bash_scripts/run_highest_mock_proba.sh data/VERO-1/embeddings_and_metadata.csv VERO-1/ $3
sbatch bash_scripts/run_highest_mock_proba.sh data/VERO-2/embeddings_and_metadata.csv VERO-2/ $4

