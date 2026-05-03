import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from datetime import datetime
from PIL import Image
import json

# Import the new image processor
from image_processor import DigitImageProcessor

# Initialize Flask app
app = Flask(__name__, template_folder='Templates', static_folder='Static')
CORS(app)

# Configuration
MODEL_PATH = 'model_cnn.keras'  # Use new CNN model
MODEL_INPUT_SIZE = 28

# Load the CNN model
try:
    model = load_model(MODEL_PATH)
    print('[OK] CNN Model loaded successfully')
    print(f'  Model input shape: {model.input_shape}')
    print(f'  Model output shape: {model.output_shape}')
except FileNotFoundError:
    print(f'[ERROR] Model file not found: {MODEL_PATH}')
    print('  Make sure model_cnn.keras has been trained using train_cnn_model.py')
    model = None
except Exception as e:
    print(f'[ERROR] Error loading model: {e}')
    import traceback
    traceback.print_exc()
    model = None

# Initialize image processor
processor = DigitImageProcessor(debug=True)

# Create required folders if they don't exist
os.makedirs('final_processed_images', exist_ok=True)
os.makedirs('mobile_uploads', exist_ok=True)

# ============================================
# ROUTES
# ============================================

def save_final_processed_image(processed_image, source_type, filename=None):
    """
    Save final processed image with metadata
    Args:
        processed_image: 28x28 normalized image (float 0-1)
        source_type: 'canvas' or 'mobile'
        filename: original filename for mobile uploads
    Returns:
        metadata dict with image path and info
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # With milliseconds
    
    # Convert normalized image to uint8 for saving
    img_uint8 = (processed_image * 255).astype(np.uint8)
    
    # Create filename
    if filename:
        name_part = filename.rsplit('.', 1)[0]
        final_filename = f"{timestamp}_{source_type}_{name_part}.png"
    else:
        final_filename = f"{timestamp}_{source_type}_drawing.png"
    
    final_path = os.path.join('final_processed_images', final_filename)
    
    # Save image
    cv2.imwrite(final_path, img_uint8)
    
    return final_filename, timestamp

@app.route('/')
def index():
    '''Serve the main page'''
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    API endpoint for prediction
    Accepts: Canvas drawing (base64)
    '''
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        print(f"\n{'='*70}")
        print('[PREDICT] Canvas Drawing Input')
        print(f"{'='*70}")
        
        # Process image using the pipeline
        # IMPORTANT: Canvas drawings are already clean, NO denoising needed!
        processed_image = processor.process(
            data['image'],
            input_type='base64',
            denoise_method=None,  # Don't denoise clean canvas drawings!
            debug=True
        )
        
        # Save final processed image
        final_filename, timestamp = save_final_processed_image(processed_image, 'canvas')
        
        # Add batch dimension: (28, 28) -> (1, 28, 28, 1)
        processed_image_batch = np.expand_dims(processed_image, axis=-1)
        processed_image_batch = np.expand_dims(processed_image_batch, axis=0)
        
        print(f'[DEBUG] Input to model shape: {processed_image_batch.shape}')
        
        # Make prediction
        prediction = model.predict(processed_image_batch, verbose=0)
        
        # Get the predicted digit and confidence
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        print(f"\n{'='*70}")
        print(f'[RESULT] Predicted Digit: {predicted_digit}')
        print(f'[CONFIDENCE] {confidence:.4f} ({confidence*100:.2f}%)')
        print(f"{'='*70}")
        print(f'\n[PROBABILITIES] All digits:')
        for i, prob in enumerate(prediction[0]):
            bar_length = int(prob * 40)
            bar = '|' * bar_length
            print(f'  Digit {i}: {prob:.4f} {bar}')
        print(f"\n{'='*70}\n")
        print(f'[SAVED] Final processed image: final_processed_images/{final_filename}')
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'source': 'canvas',
            'processed_image': final_filename,
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        }
        metadata_path = os.path.join('final_processed_images', f"{timestamp}_canvas_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': prediction[0].tolist(),
            'processed_image': final_filename,
            'metadata': metadata_path
        })
    
    except Exception as e:
        print(f'[ERROR] {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/predict-image', methods=['POST'])
def predict_image():
    '''
    API endpoint for image upload prediction
    Accepts: Image file (PNG, JPG, BMP, WEBP, etc.)
    '''
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'}
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported format. Allowed: {", ".join(allowed_extensions)}'
            }), 400
        
        print(f"\n{'='*70}")
        print(f'[PREDICT] Image Upload - {file.filename}')
        print(f"{'='*70}")
        
        # Read file
        file_bytes = file.read()
        
        # Save original uploaded file
        upload_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        original_filename = f"{upload_timestamp}_original_{file.filename}"
        original_path = os.path.join('mobile_uploads', original_filename)
        with open(original_path, 'wb') as f:
            f.write(file_bytes)
        print(f'[INFO] Original uploaded file saved to: {original_path}')
        
        # Check actual pixel dimensions first
        pil_check = Image.open(io.BytesIO(file_bytes)).convert('L')
        img_w, img_h = pil_check.size
        print(f'[INFO] Image dimensions: {img_w}x{img_h}')

        if img_w == 28 and img_h == 28:
            # Already 28x28 — send directly to model, no processing needed
            print('[INFO] Already 28x28 -> sending directly to model')
            img_arr = np.array(pil_check, dtype=np.float32) / 255.0
            
            # Check if image needs inversion (model expects white digit on black bg)
            # If mean pixel value is high (>0.5), it means mostly white background
            if np.mean(img_arr) > 0.5:
                print('[INFO] Light background detected on 28x28 image -> inverting colors')
                img_arr = 1.0 - img_arr
                
            processed_image = img_arr
        else:
            # Run full v2 pipeline: adaptive threshold + contour detection
            print('[INFO] Not 28x28 -> running preprocessing pipeline')
            processed_image = processor.process(
                file_bytes,
                input_type='bytes',
                debug=True
            )
        
        # Save final processed image
        final_filename, timestamp = save_final_processed_image(processed_image, 'mobile', file.filename)
        
        # Add batch dimension: (28, 28) -> (1, 28, 28, 1)
        processed_image_batch = np.expand_dims(processed_image, axis=-1)
        processed_image_batch = np.expand_dims(processed_image_batch, axis=0)
        
        print(f'[DEBUG] Input to model shape: {processed_image_batch.shape}')
        
        # Make prediction
        prediction = model.predict(processed_image_batch, verbose=0)
        
        # Get the predicted digit and confidence
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        all_probabilities = prediction[0].tolist()
        
        print(f"\n{'='*70}")
        print(f'[RESULT] Predicted Digit: {predicted_digit}')
        print(f'[CONFIDENCE] {confidence:.4f} ({confidence*100:.2f}%)')
        print(f"{'='*70}")
        print(f'\n[PROBABILITIES] All digits:')
        for i, prob in enumerate(all_probabilities):
            bar = '|' * int(prob * 40)
            print(f'  Digit {i}: {prob:.4f} {bar}')
        print(f"\n{'='*70}\n")
        print(f'[SAVED] Original upload: mobile_uploads/{original_filename}')
        print(f'[SAVED] Final processed: final_processed_images/{final_filename}')
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'source': 'mobile_upload',
            'original_file': original_filename,
            'processed_image': final_filename,
            'original_filename': file.filename,
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': all_probabilities
        }
        metadata_path = os.path.join('final_processed_images', f"{timestamp}_mobile_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': all_probabilities,
            'filename': file.filename,
            'original_upload': original_filename,
            'processed_image': final_filename,
            'metadata': metadata_path
        })
    
    except Exception as e:
        print(f'[ERROR] {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    '''Health check endpoint'''
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_type': 'CNN',
        'input_size': MODEL_INPUT_SIZE,
        'expected_accuracy': '99.36%'
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    '''Get model information'''
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model': 'CNN Digit Recognition',
        'framework': 'TensorFlow/Keras',
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'parameters': model.count_params(),
        'accuracy_on_mnist': 0.9936,
        'supported_inputs': ['canvas_drawing', 'image_upload'],
        'supported_formats': ['PNG', 'JPG', 'JPEG', 'WEBP'],
        'preprocessing': [
            'Denoise (bilateral/morphological)',
            'Binarization (Otsu threshold)',
            'Digit extraction',
            'Resize & pad to 28x28',
            'Normalize 0-1',
            'Inversion if needed'
        ]
    })

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print('\n' + '='*80)
    print('Professional Digit Recognition System')
    print('='*80)
    print(f'\nModel: CNN with 99.36% accuracy')
    print(f'Framework: TensorFlow/Keras')
    
    if model:
        print('\n[OK] Backend ready to serve predictions')
        print(f'\nEndpoints:')
        print(f'  Frontend: http://localhost:7860')
        print(f'  Canvas Predict: POST http://localhost:7860/predict')
        print(f'  Image Predict: POST http://localhost:7860/predict-image')
        print(f'  Health Check: http://localhost:7860/health')
        print(f'  Model Info: http://localhost:7860/model-info')
    else:
        print('\n[X] ERROR: Model not loaded!')
        print('  Please run: python train_cnn_model.py')
    
    print(f'\nPress CTRL+C to stop the server\n')
    print('='*80 + '\n')
    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=7860)
