from utils import get_all_h_param_comb, read_data, split_data

def test_hparam_combinations():
    gammas = [0.1, 0.1, 0.005, 0.123, 0.879, 0.009]
    cparams = [1, 10, 100]

    comb = get_all_h_param_comb(gammas, cparams) 

    expected_p1 = (0.1, 1)

    assert len(comb) == len(gammas) * len(cparams)
    
    assert expected_p1 in comb 

def test_splitting_data():
    X, y = read_data()

    X = X[:100, :, :]
    y = y[:100]

    test_size = 0.1
    dev_size = 0.6
    train_size = 1 - (test_size + dev_size)

    X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(X, y, test_size=test_size, dev_size=dev_size)

    assert (len(X_train) == int(train_size * len(X))) and (len(X_test) == int(test_size * len(X))) and (len(X_dev) == int(dev_size * len(X)))