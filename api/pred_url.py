import matplotlib.pyplot as plt
import sys
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pdb
from joblib import dump,load
import numpy as np
# import skimage
# from skimage.transform import resize
import pandas as pd
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return render_template("welcome.html")

@app.route('/predict/<model>', methods=['POST'])
def compare_digits(model):
    try:
        # Get the two image files from the request
        data = request.json  # Parse JSON data from the request body
        image1 = data.get('image', [])

        # Preprocess the images and make predictions
        digit1 = predict_digit(model, image1)

        # Compare the predicted digits and return the result
        result = digit1

        return jsonify({"label" : result})
    except Exception as e:
        return jsonify({"label" : "Error"})
    
def predict_digit(model_type, image):
    try:
        model = load_model(model_type)
       # Convert the input list to a numpy array and preprocess for prediction
        img_array = np.array(image, dtype=np.float32).reshape(1, -1)

        prediction = model.predict(img_array)
        digit = int(prediction[0]) 
        return digit
    except Exception as e:
        return str(e)
    
def load_model(model_type):  
    svm = load('./models/svm_gamma:0.0001_C:1.joblib')
    tree = load('./models/tree_max_depth:5.joblib')
    lr = load("./models/M23CSA016__lr_newton-cg.joblib")

    models = {
        'svm' : svm,
        'tree' : tree,
        'lr' : lr
    }

    return models[model_type]


if __name__ == '__main__':
    app.run()
