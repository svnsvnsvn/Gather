"""
Model Conversion Utility for Web Deployment
Converts trained Keras models to TensorFlow.js format for web deployment
"""

import tensorflow as tf
import os
import json

def convert_to_tensorflowjs(model_path, output_dir, model_name="waste_classifier"):
    """
    Convert a Keras model to TensorFlow.js format
    
    Args:
        model_path: Path to the .h5 model file
        output_dir: Directory to save the converted model
        model_name: Name for the converted model
    """
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("TensorFlow.js not installed. Run: pip install tensorflowjs")
        return False
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to TensorFlow.js
    print(f"Converting to TensorFlow.js format...")
    tfjs.converters.save_keras_model(
        model, 
        output_dir,
        quantize_float16=True  # Reduces model size
    )
    
    print(f"Model converted successfully to {output_dir}")
    return True

def convert_to_tflite(model_path, output_path):
    """
    Convert a Keras model to TensorFlow Lite format for mobile deployment
    
    Args:
        model_path: Path to the .h5 model file  
        output_path: Path to save the .tflite file
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("Converting to TensorFlow Lite...")
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    
    # Print model info
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")

def create_model_metadata(class_names, model_info_path):
    """
    Create metadata file for the model
    
    Args:
        class_names: List of class names
        model_info_path: Path to save the metadata JSON
    """
    metadata = {
        "classes": class_names,
        "input_shape": [224, 224, 3],
        "model_type": "image_classification",
        "preprocessing": "mobilenet_v2"
    }
    
    with open(model_info_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved to {model_info_path}")

if __name__ == "__main__":
    # Example usage - update paths as needed
    MODEL_PATH = "../models/phase1_final_model.h5"
    WEB_OUTPUT = "../web/static/model"
    MOBILE_OUTPUT = "../models/waste_classifier.tflite"
    
    # Load class names
    try:
        with open('../models/class_names.json', 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        # Default Kaggle classes if file doesn't exist
        class_names = [
            'battery', 'biological', 'brown-glass', 'cardboard', 
            'clothes', 'green-glass', 'metal', 'paper', 
            'plastic', 'shoes', 'trash', 'white-glass'
        ]
    
    if os.path.exists(MODEL_PATH):
        print("Converting trained model for deployment...")
        
        # Convert for web deployment
        convert_to_tensorflowjs(MODEL_PATH, WEB_OUTPUT)
        
        # Convert for mobile deployment  
        convert_to_tflite(MODEL_PATH, MOBILE_OUTPUT)
        
        # Create metadata
        create_model_metadata(class_names, "../web/static/model_info.json")
        
        print("\n✅ Model conversion complete!")
        print(f"Web model: {WEB_OUTPUT}")
        print(f"Mobile model: {MOBILE_OUTPUT}")
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Train your model first using the phase1_garbage_classifier.ipynb notebook")