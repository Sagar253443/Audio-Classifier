import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the voice classifier model
model = load_model('voice_classifier.h5')

# Function to preprocess audio file and predict class
def preprocess_audio(audio_file):
    try:
        audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return None

def predict_class(audio_file):
    try:
        # Preprocess audio file
        features = preprocess_audio(audio_file)
        
        if features is None:
            raise ValueError('Error in preprocessing audio')
        
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Perform prediction using the loaded model
        class_probabilities = model.predict(features)
        
        # Get predicted class index
        predicted_class_index = np.argmax(class_probabilities, axis=1)[0]
        
        # Define class labels (replace with your actual class names)
        class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                       'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
                       'siren', 'street_music']
        
        # Get predicted class label
        predicted_class = class_names[predicted_class_index]
        
        return predicted_class
    
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Save the uploaded file
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Get predicted class
        predicted_class = predict_class(file_path)
        
        # Remove the temporary file after processing
        os.remove(file_path)
        
        return jsonify({'result': predicted_class})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
