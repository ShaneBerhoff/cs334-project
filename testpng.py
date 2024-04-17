import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

audio_path = 'Data/Audio/1001_DFA_ANG_XX.wav'
audio, sample_rate = librosa.load(audio_path, sr=None)

n_fft = 2048  # The number of data points used in each block for the FFT
hop_length = 512  # The number of samples between successive frames
n_mels = 128  # Number of Mel bands to generate

# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
# Convert to log scale (dB)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()

# Save the figure
plt.savefig('Data/Spectrogram/mel_spectrogram.png')
plt.close()