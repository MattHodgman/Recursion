# Arguments: the disease condition shap value cutoff for each experiment.

# Compute cosine similarity of treatment embeddings to controls.
sbatch run_cosine_sim.sh HRCE-1 $1
sbatch run_cosine_sim.sh HRCE-2 $2
sbatch run_cosine_sim.sh VERO-1 $3
sbatch run_cosine_sim.sh VERO-2 $4

# Compute the probability of a treatment being classified as a control.
sbatch run_highest_mock_proba.sh HRCE-1/embeddings_and_metadata.csv HRCE-1/ $1
sbatch run_highest_mock_proba.sh HRCE-2/embeddings_and_metadata.csv HRCE-2/ $2
sbatch run_highest_mock_proba.sh VERO-1/embeddings_and_metadata.csv VERO-1/ $3
sbatch run_highest_mock_proba.sh VERO-2/embeddings_and_metadata.csv VERO-2/ $4