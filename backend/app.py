from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import logging
import os
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine if we're in production and need to serve static files
PRODUCTION = os.environ.get('FLASK_ENV') == 'production'
STATIC_FOLDER = '../web-app/dist' if PRODUCTION else None

app = Flask(__name__, static_folder=STATIC_FOLDER)
CORS(app)  # Enable CORS for all routes

# Load configuration
config = Config()
model = None

def load_model():
    global model
    try:
        model_path = str(config.model_path)
        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Input shape: {model.input_shape}")
        logger.info(f"Output shape: {model.output_shape}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def preprocess_image(image_data):
    """Preprocess image for MobileNetV2 model"""
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            # Remove data:image/jpeg;base64, prefix
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size from config
        target_size = config.target_size
        image = image.resize(target_size)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Add batch dimension and normalize
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32')
        
        # MobileNetV2 preprocessing: normalize to [-1, 1]
        img_array = (img_array / 127.5) - 1
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        img_array = preprocess_image(data['image'])
        if img_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = config.class_names[predicted_class_idx]
        
        # Get all class probabilities
        all_predictions = []
        for i, prob in enumerate(predictions[0]):
            all_predictions.append({
                'class': config.class_names[i],
                'confidence': float(prob)
            })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': config.class_names})

# Serve React app in production
if PRODUCTION:
    @app.route('/')
    def serve_react_app():
        return send_from_directory(app.static_folder, 'index.html')
    
    @app.route('/<path:path>')
    def serve_static_files(path):
        # Try to serve the file, if not found serve React app (for client-side routing)
        try:
            return send_from_directory(app.static_folder, path)
        except:
            return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    logger.info("Starting Waste Classifier API...")
    if load_model():
        logger.info("Model loaded successfully! Starting server...")
        # Use environment variables for production deployment
        import os
        # For Railway deployment, always bind to 0.0.0.0 in production
        host = '0.0.0.0' if os.environ.get('FLASK_ENV') == 'production' else os.environ.get('HOST', config.api_host)
        port = int(os.environ.get('PORT', config.api_port))
        debug = os.environ.get('FLASK_ENV') != 'production'
        
        logger.info(f"Starting server on {host}:{port} (debug={debug})")
        
        app.run(
            debug=debug, 
            host=host, 
            port=port
        )
    else:
        logger.error("Failed to load model. Server not started.")
