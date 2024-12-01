from flask import Flask, request, render_template, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Flask app setup
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained .h5 model
MODEL_PATH = './cifar10_model.h5'
model = load_model(MODEL_PATH)

# Class labels for CIFAR-10 dataset
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def prepare_image(file_path):
    """
    Process the uploaded image to match model input dimensions (32x32x3).
    """
    # Load the image with the target size
    img = load_img(file_path, target_size=(32, 32))
    
    # Convert image to numpy array
    img_array = img_to_array(img)
    
    # Normalize the image data to [0, 1]
    img_array = img_array / 255.0
    
    # Add batch dimension (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the file part is in the request
        if 'file' not in request.files:
            return "No file part in the request", 400
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return "No selected file", 400

        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the image and make a prediction
            img_array = prepare_image(file_path)
            predictions = model.predict(img_array)
            
            # Decode the predictions
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions)

            return f"Prediction: {predicted_class} (Confidence: {confidence:.2f})"
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
