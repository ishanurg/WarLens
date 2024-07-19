from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Set the path to the model and load it
model_path = 'model_excp.h5'  # Update this to match the path where your model is saved
model = tf.keras.models.load_model(model_path)

# Set the folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels
class_labels = ['combat', 'destroyed_building', 'fire', 'humanterian', 'vehicles']

# Function to load and preprocess the image
def load_and_preprocess_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Route to display the upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route to handle the image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            app.logger.error('No file part in the request')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            app.logger.error('No file selected for uploading')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img_array = load_and_preprocess_image(filepath)
            
            # Make a prediction using the model
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])
            predicted_label = class_labels[predicted_class]
            confidence = prediction[0][predicted_class] * 100
            
            app.logger.info(f"Prediction: {predicted_label}, Confidence: {confidence}%")
            
            # Delete the uploaded image
            os.remove(filepath)
            
            return render_template('result.html', label=predicted_label, confidence=confidence)
    except Exception as e:
        # Log the error and return an error message
        app.logger.error(f"Error processing image: {e}")
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
