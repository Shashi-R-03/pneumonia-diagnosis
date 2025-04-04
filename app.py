<<<<<<< HEAD
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cohere

app = Flask(__name__)

# Path to the folder where uploaded images will be saved
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('pneumonia_model.h5')

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the Cohere client
co = cohere.Client("YOUR API KEY")  # Add your Cohere API key here

@app.route('/')
def index():
    return render_template('index.html')  # The HTML form to upload an image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save the uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Preprocess the image and make a prediction
            result = predict_pneumonia(file_path)
        except Exception as e:
            result = f"Error processing image: {str(e)}"

        return render_template('result.html', result=result, image_path=file.filename)

    return redirect(url_for('index'))

def predict_pneumonia(image_path):
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        prediction = model.predict(img_array)
        return 'Pneumonia Detected' if prediction[0] > 0.5 else 'Normal'
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message")
    try:
        response = co.chat(
            message=user_message,
            model="command",  # Use the appropriate model
            temperature=0.7
        )
        chatbot_response = response.text.strip()
        return jsonify(response=chatbot_response)
    except Exception as e:
        return jsonify(response=f"An error occurred: {str(e)}")


@app.route('/booking')
def booking():
    return render_template('booking.html')  # Render the booking page

@app.route('/result')
def result_page():
    # This route is optional if directly rendering a button in result.html
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cohere

app = Flask(__name__)

# Path to the folder where uploaded images will be saved
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('pneumonia_model.h5')

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the Cohere client
co = cohere.Client("DbqJjall1lDsQewj14wfw5uulIM7DYqKF5CsgJDA")  # Add your Cohere API key here

@app.route('/')
def index():
    return render_template('index.html')  # The HTML form to upload an image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save the uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Preprocess the image and make a prediction
            result = predict_pneumonia(file_path)
        except Exception as e:
            result = f"Error processing image: {str(e)}"

        return render_template('result.html', result=result, image_path=file.filename)

    return redirect(url_for('index'))

def predict_pneumonia(image_path):
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        prediction = model.predict(img_array)
        return 'Pneumonia Detected' if prediction[0] > 0.5 else 'Normal'
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message")
    try:
        response = co.generate(
            model="command-xlarge-nightly",  # Choose the model (trial users may have limited access)
            prompt=f"You are an AI chatbot specialized in pneumonia. {user_message}",
            max_tokens=50,
            temperature=0.7
        )
        chatbot_response = response.generations[0].text.strip()
        return jsonify(response=chatbot_response)
    except Exception as e:
        return jsonify(response=f"An error occurred: {str(e)}")

@app.route('/booking')
def booking():
    return render_template('booking.html')  # Render the booking page

@app.route('/result')
def result_page():
    # This route is optional if directly rendering a button in result.html
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> 1f86cd2daff1e128d88bc76a9e60c213cb10331e
