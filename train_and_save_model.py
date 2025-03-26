import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

# ✅ Set dataset path (ensure it is correct)
DATASET_DIR = "C:/Users/narah/OneDrive/Desktop/FINAL/archive/audio_speech_actors_01-24/"

# ✅ Map RAVDESS emotion labels
EMOTION_LABELS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def get_emotion_from_filename(filename):
    try:
        emotion_code = filename.split('-')[2]
        return EMOTION_LABELS.get(emotion_code, 'unknown')
    except Exception as e:
        print(f"❌ Error parsing filename: {e}")
        return 'unknown'

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        # Normalize and trim silence
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y, top_db=30)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)

        # Add RMS and Spectral Entropy
        rms = librosa.feature.rms(y=y).mean()
        entropy = np.mean(librosa.feature.spectral_flatness(y=y))

        # Additional features
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

        # Combine features
        combined_features = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1),
            np.mean(flatness, axis=1),
            rms, entropy, zcr, rolloff, bandwidth
        ))

        return combined_features

    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

# ✅ Load dataset and extract features
data, labels = [], []

for actor_folder in os.listdir(DATASET_DIR):
    actor_path = os.path.join(DATASET_DIR, actor_folder)

    if os.path.isdir(actor_path):
        print(f"Processing: {actor_path}")

        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                emotion = get_emotion_from_filename(file)

                features = extract_features(file_path)
                if features is not None:
                    data.append(features)
                    labels.append(emotion)

if len(data) == 0:
    print("❌ No audio files found! Check dataset path.")
    exit()

X = np.array(data)
y = np.array(labels)

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ✅ Handle class imbalance using SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

# ✅ Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Optimize model with RandomizedSearchCV
params = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [30, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}

xgb_model = XGBClassifier(random_state=42)
search = RandomizedSearchCV(xgb_model, params, n_iter=20, cv=5, n_jobs=-1, verbose=2)
search.fit(X_train, y_train)

# ✅ Evaluate model
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("🔍 Model Performance:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ✅ Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
