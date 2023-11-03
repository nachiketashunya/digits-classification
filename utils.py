# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm, tree
from sklearn.model_selection import train_test_split
import itertools
import pdb
from joblib import dump, load

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images


def read_data():
    digits = datasets.load_digits()
    
    return digits.images, digits.target 

def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))

    return data

def split_data(X, y, test_size, dev_size, random_state=1):
     # Split data into test and temporary (train + dev) sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Calculate the ratio between dev and temp sizes
    dev_ratio = dev_size / (1 - test_size)
    
    # Split temporary data into train and dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=dev_ratio, shuffle=False)
    
    return X_train, X_test, X_dev, y_train, y_test, y_dev

def train_model(X, y, model_params, model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC 
    
    if model_type == "tree":
        clf = tree.DecisionTreeClassifier
    
    model = clf(**model_params) 

    # pdb.set_trace()

    # Learn the digits on the train subset
    model.fit(X, y)

    return model 

def get_all_h_param_comb(gamma_list,c_list):
    return list(itertools.product(gamma_list, c_list))

def get_combinations(param_name, param_values, base_combinations):    
    new_combinations = []
    for value in param_values:
        for combination in base_combinations:
            combination[param_name] = value
            new_combinations.append(combination.copy())    
    return new_combinations

def get_hyperparam_comb(dict_of_param_list):
    base_comb = [{}]
    for p_name, p_values in dict_of_param_list.items():
        base_comb = get_combinations(p_name, p_values, base_comb)
    return base_comb

def hparams_tune(X_train, X_dev, y_train, y_dev, params, model_type):
    best_accur_sofar = -1
    best_model_path = ""

    for h_params in params:
        cur_model = train_model(X_train, y_train, h_params, model_type=model_type)
        # Predict the value of the digit on the test subset
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)

        if cur_accuracy > best_accur_sofar:
            best_accur_sofar = cur_accuracy
            best_hparam = h_params  
            best_model_path = "./models/{}_".format(model_type) +"_".join(["{}:{}".format(k,v) for k,v in h_params.items()]) + ".joblib"
            best_model = cur_model

    # Save the best model
    dump(best_model, best_model_path) 

    return best_hparam, best_model_path, best_accur_sofar


def predict_and_eval(model, X_test, y_test):
    # Predict the value of the digit on the test subset
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, predicted)

    # ###############################################################################
    # # Below we visualize the first 4 test samples and show their predicted
    # # digit value in the title.

    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, prediction in zip(axes, X_test, predicted):
    #     ax.set_axis_off()
    #     image = image.reshape(8, 8)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title(f"Prediction: {prediction}")

    # ###############################################################################
    # # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # # the main classification metrics.

    # print(
    #     f"Classification report for classifier {model}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n"
    # )

    # ###############################################################################
    # # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # # true digit values and the predicted digit values.

    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # plt.show()

    # ###############################################################################
    # # If the results from evaluating a classifier are stored in the form of a
    # # :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
    # # `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
    # # as follows:


    # # The ground truth and predicted lists
    # y_true = []
    # y_pred = []
    # cm = disp.confusion_matrix

    # # For each cell in the confusion matrix, add the corresponding ground truths
    # # and predictions to the lists
    # for gt in range(len(cm)):
    #     for pred in range(len(cm)):
    #         y_true += [gt] * cm[gt][pred]
    #         y_pred += [pred] * cm[gt][pred]

    # print(
    #     "Classification report rebuilt from confusion matrix:\n"
    #     f"{metrics.classification_report(y_true, y_pred)}\n"
    # )   