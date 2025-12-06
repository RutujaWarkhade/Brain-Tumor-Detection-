# app.py
import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# ====== CONFIGURATION ======
BASE_DIR = r"D:\Brain_tumor"
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = 128  # must match training IMAGE_SIZE

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Labels (must match training)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load model
print("Loading model... Please wait.")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB

# ====== UTILITY FUNCTIONS ======
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path, image_size=IMAGE_SIZE):
    img = load_img(path, target_size=(image_size, image_size))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ====== ROUTES ======
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part received"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Preprocess image
            img_input = preprocess_image(filepath)

            # Predict
            preds = model.predict(img_input)
            predicted_index = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds))
            label = class_labels[predicted_index]

            # Tumor or No Tumor
            if label == "notumor":
                result_text = "No Tumor Detected"
            else:
                result_text = f"Tumor Detected: {label}"

            return render_template("index.html", 
                                   filename=filename, 
                                   result_text=result_text, 
                                   confidence=round(confidence * 100, 2))
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route("/display/<filename>")
def display_image(filename):
    return render_template("index.html", filename=filename)

# ====== MAIN ======
if __name__ == "__main__":
    app.run(debug=True)
