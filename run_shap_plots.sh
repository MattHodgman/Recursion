# For each experiment, plot the shap values for predicting plate and disease condition for each feature.
source env/bin/activate
python3 1c_shap_shap_plot.py -e HRCE-1 -p HRCE-1/shap_values_plate.csv -d HRCE-1/shap_values_disease_condition.csv -o HRCE-1 -n shap_v_shap.png
python3 1c_shap_shap_plot.py -e HRCE-2 -p HRCE-2/shap_values_plate.csv -d HRCE-2/shap_values_disease_condition.csv -o HRCE-2 -n shap_v_shap.png
python3 1c_shap_shap_plot.py -e VERO-1 -p VERO-1/shap_values_plate.csv -d VERO-1/shap_values_disease_condition.csv -o VERO-1 -n shap_v_shap.png
python3 1c_shap_shap_plot.py -e VERO-2 -p VERO-2/shap_values_plate.csv -d VERO-2/shap_values_disease_condition.csv -o VERO-2 -n shap_v_shap.png