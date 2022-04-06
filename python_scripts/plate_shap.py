# This script calculates the SHAP value for each deep learning embedding feature for predicting plate.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import pickle
import shap
import argparse
import constants

def parse_args():
    """
    Defines arguments.
    
    Returns:
        args: some sort of argparse object that the arguments can be extracted from.
    """
    parser = argparse.ArgumentParser(description='Use a random grid search a random forest model to classify plate and then calculate the SHAP value for each feature.')
    parser.add_argument('-i', '--input', help='CSV of DL embeddings and metadata for a specific experiment.', type=str, required=True)
    parser.add_argument('-o', '--output', help='Directory to write output to. Default is current directory.', type=str, default='.', required=False)
    parser.add_argument('-n', '--outputFileName', help='Name of the ouput file.', type=str, default='shap_values_disease_condition.csv', required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Main."""

    # Parse arguments.
    args = parse_args()
    out_dir = args.output.rstrip('/')
    out_file_name = args.outputFileName

    # Load data and preprocess data.
    df = pd.read_csv(args.input)

    feature_cols = [col for col in df.columns if constants.FEATURE_PREFIX in col]
    # subset to disease_condition=="" or "Mock"
    df=df[(df[constants.DISEASE_CONDITION].isna()) | (df[constants.DISEASE_CONDITION]==constants.MOCK)]
    e = df.experiment.unique().tolist()[0]
    
    # Split data to train on the controls and test on the experiments with drugs
    y = df[constants.PLATE].astype('category')
    X = df[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)

    del df
    
    # Random Grid Search
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1100, num = 50)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                 'bootstrap': bootstrap}
    # set up model
    clf = RandomForestClassifier()
    clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = constants.CV, verbose=2, random_state=constants.RANDOM_STATE, n_jobs = -1)
    # Fit the random search model
    clf_random.fit(X_train, y_train)
    # get best results
    print("Best hyperparameters: " + str(clf_random.best_params_))
    # save best parameters dictionary
    print(e + " writing pickle file (Pickle Rick!)")
    
    file_name=e+'_plate_RF.pickle'
    with open(file_name, 'wb') as config_dictionary_file:
        pickle.dump(clf_random.best_params_, config_dictionary_file)
    
    
    # fit RF
    clf = RandomForestClassifier(n_estimators=clf_random.best_params_['n_estimators'], 
                                 min_samples_split = clf_random.best_params_['min_samples_split'], 
                                 min_samples_leaf = clf_random.best_params_['min_samples_leaf'],
                                 max_features = clf_random.best_params_['max_features'], 
                                 max_depth=clf_random.best_params_['max_depth'], 
                                 bootstrap = clf_random.best_params_['bootstrap'],
                                 random_state=constants.RANDOM_STATE).fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    # get performance metrics from test set
    targets = [str(int) for int in list(set(y_test))]
    print(e)
    print(classification_report(y_test, y_preds, target_names=targets))
    # Get shap values and output .csv
    print("Computing SHAP values")
    # Calculate shap values.
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    # Write all shap values.
    mean_feature_shap_values = np.abs(shap_values).mean(0).mean(0)
    df_shap = pd.DataFrame(mean_feature_shap_values, columns=[constants.SHAP_VALUE])
    df_shap[constants.FEATURE] = constants.FEATURE_PREFIX + df_shap.index.astype(str)
    df_shap = df_shap[[constants.FEATURE, constants.SHAP_VALUE]]
    df_shap = df_shap.sort_values(by=constants.SHAP_VALUE, ascending=False)
    df_shap.to_csv(f'{out_dir}/{out_file_name}', index=False)