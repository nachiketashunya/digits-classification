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


# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)


# Load the data
# 1
X, y = read_data()


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

test_sizes = [0.2]
dev_sizes = [0.2]

num_runs = 5
results = []

for run_i in range(5):
    result_dict = {}
    for ts in test_sizes:
        for ds in dev_sizes:
            X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(X, y, test_size=ts, dev_size=ds)

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
                print(f"Train Size : {1 - (ts+ds)} Test Size : {ts} Dev Size : {ds}")
                print(f"Train Accuracy : {train_accuracy:.02f} Dev Accuracy : {cur_accur_sofar:.02f} Test Accuracy : {test_accuracy:.02f}") 
                print("\n")

                cur_run_results = {'model_type': model_type, 'run_index': run_i, 'train_acc':train_accuracy, 'test_acc': test_accuracy, 'dev_acc': cur_accur_sofar}
                results.append(cur_run_results)


print(pd.DataFrame(results).groupby('model_type').describe().T)
