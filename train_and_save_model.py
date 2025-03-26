import numpy as np           # For array manipulation
import librosa
# Replace old function with updated one
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        # ✅ Normalize audio to a consistent volume level
        y = librosa.util.normalize(y)

        # ✅ Trim silence with a stable threshold (prevents over-trimming)
        y, _ = librosa.effects.trim(y, top_db=30)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Additional features
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

        # Combine extracted features
        combined_features = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1),
            zcr, rolloff, bandwidth
        ))

        print(f"🔍 Extracted features shape: {combined_features.shape}")
        return combined_features

    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None
