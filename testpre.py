from src.preprocessing import AudioUtil
import matplotlib.pyplot as plt

data_path = "Data/archive/Crema"
duration_ms = 2618
sr = 24414
channel = 1
shift_pct = 0.3
hop = int((sr*(duration_ms/1000))//(224-1)-5) # magic formula

audio_file = "Data/archive/Crema/1001_DFA_HAP_XX.wav"

aud = AudioUtil.load_audio(audio_file)
reaud = AudioUtil.resample(aud, sr)
rechan = AudioUtil.rechannel(reaud, channel)
dur_aud = AudioUtil.pad_trunc(rechan, duration_ms, padding_type='zero')
shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
sgram = AudioUtil.spectro_gram(shift_aud, n_mels=224, n_fft=2048, hop_len=hop)
print(sgram.shape)
fig, axs = plt.subplots(1, 1, figsize=(10, 4))
axs.imshow(sgram[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
axs.set_title('Mel Spectrogram (dB)')
axs.set_ylabel('Mel Frequency Bin')
axs.set_xlabel('Time Window')
plt.colorbar(axs.imshow(sgram[0].numpy(), origin='lower', aspect='auto', cmap='viridis'), ax=axs)
plt.savefig('Data/spectrogram/mel_spectrogram.png')
plt.close()