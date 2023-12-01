from api.pred_url import app
import numpy as np
from flask import jsonify
import json
from sklearn import datasets
import pytest

digits = datasets.load_digits()

def test_digit_0():
    digit_to_test = 0
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test

def test_digit_1():
    digit_to_test = 1
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test

def test_digit_2():
    digit_to_test = 2
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test

def test_digit_3():
    digit_to_test = 3
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test


def test_digit_4():
    digit_to_test = 4
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test


def test_digit_5():
    digit_to_test = 5
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test

def test_digit_6():
    digit_to_test = 6
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test

def test_digit_7():
    digit_to_test = 7
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test

def test_digit_8():
    digit_to_test = 8
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test

def test_digit_9():
    digit_to_test = 9
    indices = np.where(digits.target == digit_to_test)[0]
    random_index = indices[0]
    sample_data = digits.data[random_index]
    sample_data_list = sample_data.tolist()

    response = app.test_client().post("/digitpredict", json={"image": sample_data_list})

    assert response.status_code == 200
    predicted_label = response.get_json()['label']
    assert predicted_label == digit_to_test
