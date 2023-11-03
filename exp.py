"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# Importing required functions from utils.py
from utils import preprocess_data, split_data, read_data, train_model, predict_and_eval, hparams_tune, get_hyperparam_comb
import pdb
from joblib import load 
import pandas as pd
import argparse

# Take values from CLI
# Create ArgumentParser object
# parser = argparse.ArgumentParser(description='Description of your program')

# # Add arguments
# parser.add_argument('--num_runs', type=float, help='Nums of runs for model training', default=5)
# parser.add_argument('--test_size', type=float, help='Value for test size', default=0.2)
# parser.add_argument('--dev_size', type=float, help='Value for dev size', default=0.2)
# parser.add_argument('--model_type', choices=['svm', 'dtree'], type=str, help='Type of the model', default='svm')

# # Parse the command-line arguments
# args = parser.parse_args()

# # Access the values of the arguments
# num_runs = args.num_runs
# test_size = args.test_size
# dev_size = args.dev_size
# model_type = args.model_type


# Load the data
# 1
X, y = read_data()

num_runs = 5
test_size = 0.2
dev_size = 0.2
# Hyperparameters Combination
# 2.1 SVM
 
classifier_param_dict = {}
gammas = [0.0001, 0.001, 0.0005, 0.123, 0.1, 1]
cparams = [1, 10, 100, 1000]

p_comb_svm = {
    'gamma' : gammas,
    'C' : cparams
}

classifier_param_dict['svm'] = get_hyperparam_comb(p_comb_svm)

# 2.2 Decision Tree

max_depth_list = [5, 10, 15, 20, 50, 100]
p_comb_tree = {
    'max_depth': max_depth_list
}

classifier_param_dict['tree'] = get_hyperparam_comb(p_comb_tree)


results = []

for run_i in range(5):
    result_dict = {}

    X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(X, y, test_size=test_size, dev_size=dev_size)

    X_train = preprocess_data(X_train) 
    X_test = preprocess_data(X_test) 
    X_dev = preprocess_data(X_dev)
    y_dev = preprocess_data(y_dev)

    for model_type in classifier_param_dict:
        p_comb = classifier_param_dict[model_type]

        cur_hparam, cur_model_path, cur_accur_sofar = hparams_tune(X_train, X_dev, y_train, y_dev, p_comb, model_type)

        # Get the test accuracy 
        cur_model = load(cur_model_path)
        train_accuracy = predict_and_eval(cur_model, X_train, y_train)
        test_accuracy = predict_and_eval(cur_model, X_test, y_test)

        print(f"Model Type : {model_type}")
        print(f"Train Size : {1 - (test_size+dev_size)} Test Size : {test_size} Dev Size : {dev_size}")
        print(f"Train Accuracy : {train_accuracy:.02f} Dev Accuracy : {cur_accur_sofar:.02f} Test Accuracy : {test_accuracy:.02f}") 
        print("\n")

        cur_run_results = {'model_type': model_type, 'run_index': run_i, 'train_acc':train_accuracy, 'test_acc': test_accuracy, 'dev_acc': cur_accur_sofar}
        results.append(cur_run_results)


print(pd.DataFrame(results).groupby('model_type').describe().T)
