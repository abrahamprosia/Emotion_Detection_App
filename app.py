from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('emotion_detection_model.keras')

# Load class names (you should provide this list or load it from a file)
class_names = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]


# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224), normalization_factor=255.0):
    # Resize the image
    image = image.resize(target_size)
    # Convert to numpy array and normalize
    image_array = np.array(image) / normalization_factor
    # Add batch dimension
    image_tensor = np.expand_dims(image_array, axis=0)
    return image_tensor


# Function to predict the class of the image
def predict_image(model, img_array, class_names):
    # Get model predictions (probabilities)
    predictions = model.predict(img_array)

    # Get the predicted class label (index of the highest probability)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_idx]

    return predicted_class_name, predictions[0][predicted_class_idx]


@app.route('/')
def index():
    # Render an HTML page for the user to upload an image
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return render_template('result.html', error="No file found")

    try:
        # Get the uploaded image
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict the class
        predicted_class_name, predicted_prob = predict_image(model, processed_image, class_names)

        # Optional: Encode the image for display in the response
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Render the result template with the prediction
        return render_template(
            'result.html',
            predicted_class=predicted_class_name,
            probability=f"{predicted_prob:.4f}",
            image_data=encoded_image
        )

    except Exception as e:
        return render_template('result.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
