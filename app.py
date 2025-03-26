from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import pickle
import sounddevice as sd
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and scaler
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("✅ Model and scaler loaded successfully!")
else:
    model, scaler = None, None
    print("❌ Model or scaler not loaded. Check the path or train the model again.")

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/record_and_predict', methods=['GET', 'POST'])
def record_and_predict():
    if request.method == 'POST':
        # Record and save audio
        audio_path = os.path.join(UPLOAD_FOLDER, "user_input.wav")
        fs = 44100
        duration = 5
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        sf.write(audio_path, recording, fs)

        # Extract features and predict emotion
        try:
            features = extract_features(audio_path)
            print(f"Extracted features shape: {features.shape}")  # Debug output

            features_scaled = scaler.transform([features])
            print(f"Scaled features shape: {features_scaled.shape}")  # Debug output

            prediction = model.predict(features_scaled)[0]
            predicted_emotion = prediction
        except Exception as e:
            predicted_emotion = f"⚠️ Error: {str(e)}"

        return render_template('result.html', emotion=predicted_emotion)

    return render_template('record.html')

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Extract MFCCs, Chroma, Mel Spectrogram, Spectral Contrast (160 features)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Combine features into a 160-dimensional array
    combined_features = np.hstack((
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1),
        np.mean(contrast, axis=1)
    ))

    return combined_features

if __name__ == "__main__":
    app.run(debug=True)