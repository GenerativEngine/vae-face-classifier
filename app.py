import os
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
import joblib
from PIL import Image
import io
import base64

app = Flask(__name__)

# Define the path where models are expected
MODEL_DIR = './models' # Assuming models are in a 'models' directory relative to app.py

# Load models globally when the app starts
try:
    encoder = load_model(os.path.join(MODEL_DIR, 'encoder_model.h5'))
    svm_classifier = joblib.load(os.path.join(MODEL_DIR, 'svm_classifier.pkl'))
    target_names = joblib.load(os.path.join(MODEL_DIR, 'target_names.pkl'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure 'encoder_model.h5', 'svm_classifier.pkl', and 'target_names.pkl' are in the 'models' directory.")
    encoder = None
    svm_classifier = None
    target_names = None

@app.route('/')
def home():
    return "Face Recognition API is running! Send POST requests to /predict."

@app.route('/predict', methods=['POST'])
def predict():
    if encoder is None or svm_classifier is None or target_names is None:
        return jsonify({"error": "Models not loaded. Please check server logs."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the image file
        img = Image.open(io.BytesIO(file.read())).convert('L') # Convert to grayscale
        # Resize to the expected dimensions (h, w) from the original script
        # Assuming original_dim = h * w, and resize=0.4 was used for LFW
        # LFW images are 250x250, resize=0.4 makes them 100x100
        # So, h=100, w=100, original_dim = 10000
        # This needs to match the input shape of your encoder model
        # For LFW with resize=0.4, the dimensions are 50x37 pixels (h, w)
        # original_dim = 50 * 37 = 1850
        # Let's assume the model expects 50x37 for now, or you'll need to adapt
        # to the specific h, w from your original script's lfw_people.images.shape
        # For simplicity, let's assume the input image needs to be flattened to original_dim
        # The original script uses h, w from lfw_people.images.shape after resize=0.4
        # So, we should get these values from the training script or make them configurable.
        # For LFW with resize=0.4, h=50, w=37.
        expected_h = 50
        expected_w = 37
        img = img.resize((expected_w, expected_h)) # PIL.Image.resize expects (width, height)

        # Flatten the image and normalize
        img_array = np.array(img).flatten().astype('float32') / 255.0
        # Reshape for the encoder (batch_size, original_dim)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # Get latent space representation
        latent_space_representation = encoder.predict(img_array)

        # Predict using SVM
        prediction_index = svm_classifier.predict(latent_space_representation)[0]
        predicted_name = target_names[prediction_index]

        return jsonify({"predicted_person": predicted_name}), 200

    except Exception as e:
        return jsonify({"error": f"Processing error: {e}"}), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Run Flask app
    app.run(host='0.0.0.0', port=5000)
