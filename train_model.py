import os
import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define dataset path and emotion mapping
DATASET_DIR = "C:/Users/narah/OneDrive/Desktop/FINAL/archive/audio_speech_actors_01-24/"
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract emotion from filename (based on RAVDESS naming convention)
def get_emotion_from_filename(filename):
    emotion_code = filename.split('-')[2]  # Extract emotion code from filename
    return EMOTION_LABELS.get(emotion_code, 'unknown')  # Map the code to the emotion label

# Function to extract audio features (MFCC, Chroma, Mel Spectrogram, Spectral Contrast)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        # Extract MFCCs, Chroma, Mel Spectrogram, and Spectral Contrast
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Combine all features into a single array (160 dimensions)
        combined_features = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1)
        ))
        return combined_features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset and extract features
data, labels = [], []
for actor_folder in os.listdir(DATASET_DIR):
    actor_path = os.path.join(DATASET_DIR, actor_folder)
    if os.path.isdir(actor_path):  # Ensure it’s a folder
        print(f"Processing files in: {actor_path}")
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                emotion = get_emotion_from_filename(file)  # Extract emotion from filename
                features = extract_features(file_path)  # Extract 160-dimensional features
                if features is not None:  # Add valid features only
                    data.append(features)
                    labels.append(emotion)

# Verify if audio files were processed
if len(data) == 0:
    print("❌ No audio files found! Please check your dataset path.")
    exit()

X = np.array(data)
y = np.array(labels)

# Split dataset into training and testing sets (Stratified Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForest model with optimized hyperparameters
model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save the trained model and scaler to files
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model trained and saved successfully!")
