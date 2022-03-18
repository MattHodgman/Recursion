# Arguments: for each experiment, the shap value (from predicting disease condition) to use as a cutoff to remove features that help predict plate but not disease condition

# run sensitivity test and plot results
sbatch run_sensitivity_test.sh HRCE-1/embeddings_and_metadata.csv HRCE-1/shap_values_disease_condition.csv HRCE-1/ $1
sbatch run_sensitivity_test.sh HRCE-2/embeddings_and_metadata.csv HRCE-2/shap_values_disease_condition.csv HRCE-2/ $2
sbatch run_sensitivity_test.sh VERO-1/embeddings_and_metadata.csv VERO-1/shap_values_disease_condition.csv VERO-1/ $3
sbatch run_sensitivity_test.sh VERO-2/embeddings_and_metadata.csv VERO-2/shap_values_disease_condition.csv VERO-2/ $4