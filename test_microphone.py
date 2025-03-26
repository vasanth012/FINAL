import sounddevice as sd
import soundfile as sf

print("Recording for 5 seconds...")
fs = 44100  # Sample rate
duration = 5  # Duration in seconds

# Record audio
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
sf.write("test_audio.wav", recording, fs)
print("Recording saved as 'test_audio.wav'")
