import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
import cv2

app = Flask(__name__)

# =============================
# Create GradCAM Folder
# =============================
if not os.path.exists("static/gradcam"):
    os.makedirs("static/gradcam")

# =============================
# Database Setup
# =============================
def init_db():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patient_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        age REAL,
        gender TEXT,
        hemoglobin REAL,
        wbc REAL,
        platelets REAL,
        hematuria TEXT,
        dysuria TEXT,
        smoking TEXT,
        family_history TEXT,
        chronic_infection TEXT,
        total_images INTEGER,
        cancer_percentage REAL,
        cystitis_percentage REAL
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =============================
# Load Model
# =============================
model = tf.keras.models.load_model("bladder_model.keras")

# Extract EfficientNet from multimodal model
base_model = model.get_layer("efficientnetb1")

# =============================
# Grad-CAM Function (Image Branch Only)
# =============================
def make_gradcam_heatmap_image_branch(img_array_pp, base_model, last_conv_name="top_conv"):

    last_conv_layer = base_model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img_array_pp)
        tape.watch(conv_outputs)
        loss = tf.reduce_sum(conv_outputs)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# =============================
# Home Page
# =============================
@app.route('/')
def home():
    return render_template("index.html")

# =============================
# About Page
# =============================
@app.route('/about')
def about():
    return render_template("about.html")

# =============================
# Model Page
# =============================
@app.route('/model')
def model_page():
    return render_template("model.html")

# =============================
# Predict Page
# =============================
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():

    if request.method == 'GET':
        return render_template("predict.html")

    # Clear old GradCAM images
    for f in os.listdir("static/gradcam"):
        os.remove(os.path.join("static/gradcam", f))

    test_folder = request.form['test_folder']

    # Clinical Inputs
    age = float(request.form['age'])
    gender = float(request.form['gender'])
    hemoglobin = float(request.form['hemoglobin'])
    wbc = float(request.form['wbc'])
    platelets = float(request.form['platelets'])
    hematuria = float(request.form['hematuria'])
    dysuria = float(request.form['dysuria'])
    smoking = float(request.form['smoking'])
    family_history = float(request.form['family_history'])
    chronic_infection = float(request.form['chronic_infection'])

    clinical = np.array([[age, gender, hemoglobin, wbc,
                          platelets, hematuria, dysuria,
                          smoking, family_history, chronic_infection]])

    cancer_count = 0
    cystitis_count = 0
    total = 0
    gradcam_images = []

    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):

                img_path = os.path.join(root, file)

                # Load Image
                img_pil = Image.open(img_path).resize((256, 256))
                img = np.array(img_pil)
                img_input = np.expand_dims(img, axis=0)

                # Preprocess for EfficientNet
                img_pp = img_input.astype("float32") / 255.0

                # Multimodal Prediction
                prediction = model.predict(
                    {"image_input": img_input,
                     "clinical_input": clinical},
                    verbose=0
                )[0][0]

                # Count predictions
                if prediction > 0.5:
                    cancer_count += 1
                else:
                    cystitis_count += 1

                # =============================
                # Grad-CAM
                # =============================
                heatmap = make_gradcam_heatmap_image_branch(
                    img_pp,
                    base_model,
                    last_conv_name="top_conv"
                )

                heatmap = cv2.resize(heatmap, (256, 256))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                original = img_input[0].astype("uint8")
                superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

                output_path = os.path.join("static/gradcam", file)
                cv2.imwrite(output_path, superimposed_img)

                gradcam_images.append("gradcam/" + file)
                total += 1

    if total == 0:
        return "No images found in test folder."

    cancer_percentage = round((cancer_count / total) * 100, 2)
    cystitis_percentage = round((cystitis_count / total) * 100, 2)

    # =============================
    # Save to Database
    # =============================
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO patient_reports (
        date, age, gender, hemoglobin, wbc, platelets,
        hematuria, dysuria, smoking,
        family_history, chronic_infection,
        total_images, cancer_percentage, cystitis_percentage
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        age,
        "Male" if gender == 1 else "Female",
        hemoglobin,
        wbc,
        platelets,
        "Yes" if hematuria == 1 else "No",
        "Yes" if dysuria == 1 else "No",
        "Yes" if smoking == 1 else "No",
        "Yes" if family_history == 1 else "No",
        "Yes" if chronic_infection == 1 else "No",
        total,
        cancer_percentage,
        cystitis_percentage
    ))

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        total=total,
        cancer_percentage=cancer_percentage,
        cystitis_percentage=cystitis_percentage,
        gradcam_images=gradcam_images,
        age=age,
        gender="Male" if gender == 1 else "Female",
        hemoglobin=hemoglobin,
        wbc=wbc,
        platelets=platelets,
        hematuria="Yes" if hematuria == 1 else "No",
        dysuria="Yes" if dysuria == 1 else "No",
        smoking="Yes" if smoking == 1 else "No",
        family_history="Yes" if family_history == 1 else "No",
        chronic_infection="Yes" if chronic_infection == 1 else "No"
    )

# =============================
# Contact Page
# =============================
@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']

    print("New Message Received")
    print("Name:", name)
    print("Email:", email)
    print("Message:", message)

    return render_template("message_sent.html", name=name)

# =============================
# Run App
# =============================
if __name__ == '__main__':
    app.run(debug=True)