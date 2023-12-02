from utils import get_all_h_param_comb, read_data, split_data, preprocess_data, predict_and_eval, hparams_tune, get_hyperparam_comb
import os

def test_hparam_combinations():
    p_comb = hparams()

    comb = get_all_h_param_comb(p_comb['gamma'], p_comb['C']) 

    expected_p1 = (0.1, 1)

    assert len(comb) == len(p_comb['gamma']) * len(p_comb['C'])
    
    assert expected_p1 in comb 

# Test case for data splitting
def test_splitting_data():
    test_size = 0.1
    dev_size = 0.6
    train_size = 1 - (test_size + dev_size)
    X_train, X_dev, y_train, y_dev = get_preprocessed_data(test_size = test_size, dev_size = dev_size, train_test_split=True)

    assert (len(X_train) == int(train_size * 100)) and (len(X_dev) == int(dev_size * 100))

# Function to get hparams dictionary
def hparams():
    gammas = [0.1, 0.1, 0.005, 0.123, 0.879, 0.009]
    cparams = [1, 10, 100]

    p_comb = {
        'gamma' : gammas,
        'C' : cparams
    }

    return p_comb

# Function to get preprocessed data
def get_preprocessed_data(test_size = 0.1, dev_size = 0.7, train_test_split = False):
    X, y = read_data()

    if train_test_split:
        X = X[:100, :, :]
        y = y[:100]
        X_train, _, X_dev, y_train, _, y_dev = split_data(X, y, test_size=test_size, dev_size=dev_size)

    else:
        X_train = X[:100, :, :]
        X_dev = X[:50, :, :]
        y_train = y[:100]
        y_dev = y[:50]

    X_train = preprocess_data(X_train)  
    X_dev = preprocess_data(X_dev)
    y_train = preprocess_data(y_train)
    y_dev = preprocess_data(y_dev)

    return X_train, X_dev, y_train, y_dev 


# Test save for model saving
def test_model_saving():
    X_train, X_dev, y_train, y_dev = get_preprocessed_data()

    p_comb = [{
        'solver' : 'newton-cg'
    }]

    _, cur_model_path, _ = hparams_tune(X_train, X_dev, y_train, y_dev, p_comb, model_type="logistic")

    assert os.path.exists(cur_model_path)

