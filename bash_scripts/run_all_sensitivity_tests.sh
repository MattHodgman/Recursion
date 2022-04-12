# This script runs a sensitviity test for each experiment separately to determine how much feature dropping affects predicting disease condition.
# Arguments: for each experiment, the shap value (from predicting disease condition) to use as a cutoff to remove features that help predict plate but not disease condition

# run sensitivity test and plot results
sbatch bash_scripts/run_sensitivity_test.sh data/HRCE-1/embeddings_and_metadata.csv data/HRCE-1/shap_values_disease_condition.csv HRCE-1/ $1
sbatch bash_scripts/run_sensitivity_test.sh data/HRCE-2/embeddings_and_metadata.csv data/HRCE-2/shap_values_disease_condition.csv HRCE-2/ $2
sbatch bash_scripts/run_sensitivity_test.sh data/VERO-1/embeddings_and_metadata.csv data/VERO-1/shap_values_disease_condition.csv VERO-1/ $3
sbatch bash_scripts/run_sensitivity_test.sh data/VERO-2/embeddings_and_metadata.csv data/VERO-2/shap_values_disease_condition.csv VERO-2/ $4