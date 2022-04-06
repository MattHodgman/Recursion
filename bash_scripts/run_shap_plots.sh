# This script, for each experiment, plots the SHAP values for predicting plate and disease condition for each feature on a scatter plot.

source ../env/bin/activate
python3 ../python_scripts/shap_shap_plot.py -e HRCE-1 -p ../data/HRCE-1/shap_values_plate.csv -d ../data/HRCE-1/shap_values_disease_condition.csv -o ../data/HRCE-1 -n shap_v_shap.png
python3 ../python_scripts/shap_shap_plot.py -e HRCE-2 -p ../data/HRCE-2/shap_values_plate.csv -d ../data/HRCE-2/shap_values_disease_condition.csv -o ../data/HRCE-2 -n shap_v_shap.png
python3 ../python_scripts/shap_shap_plot.py -e VERO-1 -p ../data/VERO-1/shap_values_plate.csv -d ../data/VERO-1/shap_values_disease_condition.csv -o ../data/VERO-1 -n shap_v_shap.png
python3 ../python_scripts/shap_shap_plot.py -e VERO-2 -p ../data/VERO-2/shap_values_plate.csv -d ../data/VERO-2/shap_values_disease_condition.csv -o ../data/VERO-2 -n shap_v_shap.png