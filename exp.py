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
from utils import preprocess_data, split_data, read_data, train_model, predict_and_eval, hparams_tune
import pdb


# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)


X, y = read_data()




gammas = [0.1, 0.1, 0.005, 0.123, 0.879, 0.009]
cparams = [1, 10, 100]

p_comb = {
    'gammas' : gammas,
    'cparams' : cparams
}

test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

for ts in test_sizes:
    for ds in dev_sizes:
        X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(X, y, test_size=ts, dev_size=ds)

        X_train = preprocess_data(X_train) 
        X_test = preprocess_data(X_test) 
        X_dev = preprocess_data(X_dev)
        y_dev = preprocess_data(y_dev)

        cur_hparam, cur_model_path, cur_accur_sofar = hparams_tune(X_train, X_dev, y_train, y_dev, p_comb)

        # Get the test accuracy 
        train_accuracy = predict_and_eval(cur_model_path, X_train, y_train)
        test_accuracy = predict_and_eval(cur_model_path, X_test, y_test)

        print(f"Train Size : {1 - (ts+ds)} Test Size : {ts} Dev Size : {ds}")
        print(f"Train Accuracy : {train_accuracy:.02f} Dev Accuracy : {cur_accur_sofar:.02f} Test Accuracy : {test_accuracy:.02f}") 
        print(f"Optimal Gamma : {cur_hparam[0]} Optimal C : {cur_hparam[1]}")
        print("\n")