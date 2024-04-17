from src.preprocessing import AudioUtil
import matplotlib.pyplot as plt

data_path = "Data/archive/Crema"
duration = 4000
sr = 44100
channel = 2
shift_pct = 0.4

audio_file = "Data/archive/Crema/1001_DFA_ANG_XX.wav"

aud = AudioUtil.load_audio(audio_file)
reaud = AudioUtil.resample(aud, sr)
rechan = AudioUtil.rechannel(reaud, channel)
dur_aud = AudioUtil.pad_trunc(rechan, duration)
shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)

fig, axs = plt.subplots(1, 1, figsize=(10, 4))
axs.imshow(sgram[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
axs.set_title('Mel Spectrogram (dB)')
axs.set_ylabel('Mel Frequency Bin')
axs.set_xlabel('Time Window')
plt.colorbar(axs.imshow(sgram[0].numpy(), origin='lower', aspect='auto', cmap='viridis'), ax=axs)
plt.savefig('Data/spectrogram/mel_spectrogram.png')
plt.close()