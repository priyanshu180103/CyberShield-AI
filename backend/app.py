import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Fix CORS issues
import tensorflow as tf
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                shape=(input_shape[-1], 1),
                                initializer="glorot_uniform",
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_weights = tf.nn.softmax(tf.matmul(inputs, self.W), axis=1)
        context_vector = tf.multiply(inputs, attention_weights)
        return tf.reduce_sum(context_vector, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()

# Text cleaning function (same as in training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Load Model, Tokenizer, and Label Mapping
MODEL_PATH = "/backend/model/bilstm_model.h5"  # ✅ Update to local path
TOKENIZER_PATH = "/backend/model/tokenizer_bilstm.pkl"
LABEL_MAPPING_PATH = "/backend/model/label_mapping_bilstm.pkl"

print("🔄 Loading model, tokenizer, and label mapping...")

try:
    # Load model with AttentionLayer
    model = load_model(MODEL_PATH,custom_objects={"AttentionLayer": AttentionLayer})

    # Load tokenizer
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load label mapping
    with open(LABEL_MAPPING_PATH, 'rb') as handle:
        label_mapping = pickle.load(handle)

    # Create inverse mapping for prediction results
    inverse_mapping = {idx: label for label, idx in label_mapping.items()}

    MAX_LENGTH = 50  # ✅ Match the model's input shape

    print("✅ Model, tokenizer, and label mapping loaded successfully.")

except Exception as e:
    print(f"🚨 Error loading model/resources: {str(e)}")
    model, tokenizer, inverse_mapping = None, None, None  # Avoid breaking API


app = Flask(__name__)
CORS(app)  

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model or tokenizer not loaded"}), 500

    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        print(f"🔎 Received text: {text}")

        # Clean and tokenize input text
        cleaned_text = clean_text(text)
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')

        print(f"🔄 Padded sequence: {padded_sequences}")

        # Make prediction
        prediction_probs = model.predict(padded_sequences)[0]
        predicted_class_idx = np.argmax(prediction_probs)  # Class with highest probability
        predicted_label = inverse_mapping[predicted_class_idx]
        confidence = float(prediction_probs[predicted_class_idx]) * 100

        print(f"✅ Prediction: {predicted_label} ({confidence:.2f}% confidence)")

        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence,
            "all_probabilities": {inverse_mapping[i]: float(pred) * 100 for i, pred in enumerate(prediction_probs)}
        })

    except Exception as e:
        print(f"🚨 Prediction Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)










