import os
import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
from imblearn.over_sampling import SMOTE

# Define dataset path and emotion mapping
DATASET_DIR = "C:/Users/narah/OneDrive/Desktop/FINAL/archive/audio_speech_actors_01-24/"
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'Angry',
    '04': 'sad',
    '05': 'Happy',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract emotion from filename (based on RAVDESS naming convention)
def get_emotion_from_filename(filename):
    emotion_code = filename.split('-')[2]  # Extract emotion code from filename
    return EMOTION_LABELS.get(emotion_code, 'unknown')  # Map the code to the emotion label

# Function to extract audio features (MFCC, Chroma, Mel Spectrogram, Spectral Contrast, and more)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        # Trim silence
        y, _ = librosa.effects.trim(y)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Additional features
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

        # Combine all features into a single array
        combined_features = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1),
            zcr,
            rolloff,
            bandwidth
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
                features = extract_features(file_path)  # Extract features
                if features is not None:  # Add valid features only
                    data.append(features)
                    labels.append(emotion)

# Verify if audio files were processed
if len(data) == 0:
    print("❌ No audio files found! Please check your dataset path.")
    exit()

X = np.array(data)
y = np.array(labels)

# Check class balance
print("Emotion Distribution:", Counter(y))

# Split dataset into training and testing sets (Stratified Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForest model with optimized hyperparameters
model = RandomForestClassifier(n_estimators=1000, max_depth=40, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print("🔍 Model Evaluation on Test Data:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save the trained model and scaler to files
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model trained and saved successfully!")
