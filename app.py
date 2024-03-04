from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model  # TensorFlow is required for Keras to work
# from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2
from keras.preprocessing import image as keras_image

app = Flask(__name__)
CORS(app)  # Allow all origins by default (for development)


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()


@app.route("/hello", methods=['GET'])  # Define the GET endpoint
def hello_world():
    return "Hello, World!"


@app.route('/', methods=['POST'])  # Handle POST requests
def process_tasks():

    # Receive image file from Flutter app
    uploaded_file = request.files['image']
    
    # Choose an appropriate temporary path and save the image
    image_path = "temp_image.jpg"
    uploaded_file.save(image_path)

    # Load the image using opencv-python
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Resize the image to be at least 224x224 and then cropping from the center
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    # Normalize the image
    normalized_image_array = img.astype(np.float32) / 127.5 - 1

    # Expand dimensions to match the expected input shape
    normalized_image_array = np.expand_dims(normalized_image_array, axis=0)

    # Predict using the loaded model
    prediction = model.predict(normalized_image_array)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], "Confidence Score:", confidence_score)

    return jsonify({'class': str(class_name[2:]), 'confidence': str(confidence_score)}), 200



if __name__ == '__main__':
    app.run(port=5000, debug=True)
