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
model = load('./models/svm_gamma:0.001_C:1.joblib')

@app.route("/")
def hello_world():
    return render_template("welcome.html")

@app.route('/digitpredict', methods=['POST'])
def compare_digits():
    try:
        # Get the two image files from the request
        data = request.json  # Parse JSON data from the request body
        image1 = data.get('image', [])

        # Preprocess the images and make predictions
        digit1 = predict_digit(image1)

        # Compare the predicted digits and return the result
        result = digit1

        return jsonify({"label" : result})
    except Exception as e:
        return jsonify({"label" : "Error"})
    
def predict_digit(image):
    try:
       # Convert the input list to a numpy array and preprocess for prediction
        img_array = np.array(image, dtype=np.float32).reshape(1, -1)

        prediction = model.predict(img_array)
        digit = int(prediction[0]) 
        return digit
    except Exception as e:
        return str(e)
    
@app.route('/imagepredict', methods=['POST'])
def image_prediction():
    try:
        # Get the two image files from the request
        image1 = request.files['image1']
        image2 = request.files['image2']

        # Preprocess the images and make predictions
        digit1 = predict_image(image1)
        digit2 = predict_image(image2)

        # Compare the predicted digits and return the result
        result = {"label1": digit1, "label2": digit2}

        return jsonify(result)
    except Exception as e:
        return jsonify({"label1": "Error", "label2": "Error"})

    
def predict_image(image):
    try:
        # Preprocess the image as needed for your model
        # For example, you can use a library like PIL to resize the image
        # and convert it to grayscale before making predictions
        from PIL import Image
        img = Image.open(image).convert('L').resize((8, 8))  # Example resizing to 28x28 pixels
        img_array = np.array(img, dtype=np.float32).reshape(1, -1)

        # Make predictions using your model
        prediction = model.predict(img_array)
        digit = int(prediction[0])

        return digit
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run()
