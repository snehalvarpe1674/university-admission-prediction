from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)# Create Flask application

# Load trained model and preprocessing tools
model = load_model('model.h5')# Load the saved Keras model
scaler = joblib.load('scaler.pkl')# Load the scaler used during training
label_encoder = joblib.load('label_encoder.pkl')# Load the label encoder used for target

# Define route for home page that handles both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None # Initialize result variable
    if request.method == 'POST':  #If the form is submitted
        try:
            # Read form input
            features = [
                float(request.form.get('gre')),
                float(request.form.get('toefl')),
                float(request.form.get('university')),
                float(request.form.get('sop')),
                float(request.form.get('lor')),
                float(request.form.get('cgpa')),
                float(request.form.get('research'))
            ]
              # Scale input features using the same scaler used in training
            features_scaled = scaler.transform([features])
             # Predict admission chance using the model
            prediction = model.predict(features_scaled)[0][0]
             # Convert probability to binary classification result
            result = 'Yes' if prediction >= 0.5 else 'No'
        except Exception as e:
            result = f"Error: {e}"
             # Render the HTML template with the prediction result
    return render_template('index.html', result=result)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
