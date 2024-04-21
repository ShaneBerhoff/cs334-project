import random
import torch
import torchaudio
from torchaudio import transforms
import os
import util
import metadata

class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def load_audio(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by averaging channels
            resig = torch.mean(sig, axis=0, keepdim=True)
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        # Resample all channels at the same time 
        resampler = torchaudio.transforms.Resample(sr, newsr)
        resig = resampler(sig)

        return (resig, newsr)

    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms, padding_type='reflect'):
        sig, sr = aud
        max_len = sr//1000 * max_ms
        
        if sig.shape[-1] < max_len:
            pad_amt = (max_len - sig.shape[-1])  # Total amount of padding needed
            # Pad using reflection to avoid discontinuities typical with zero-padding
            pad = (pad_amt // 2, pad_amt - pad_amt // 2)
            
            if padding_type == 'reflect':
                sig = torch.nn.functional.pad(sig, pad, mode='reflect')
            elif padding_type == 'zero':
                sig = torch.nn.functional.pad(sig, pad, mode='constant', value=0)
            else:
                raise ValueError("Invalid padding type. Use 'zero' or 'reflect'.")
        else:
            sig = sig[:, :max_len]  # Truncate the signal if it's longer than the max length
            
        return (sig, sr)

    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None, top_db=80):
        sig,sr = aud

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

def preProcessAudio(data_path, metadata_path, save_path, sr=16000, channel=2, duration=2618, shift_pct=.03, n_mels=64, n_fft=1024, hop_len=256):
    """Process and save all audio files to tensors."""
    df = metadata.Metadata(metadata_path).getMetadata()
    os.makedirs(save_path, exist_ok=True)

    for idx, row in df.iterrows():
        audio_file = os.path.join(data_path, row['filepath'])
        class_id = row['label']

        # Audio processing pipeline
        aud = AudioUtil.load_audio(audio_file)
        reaud = AudioUtil.resample(aud, sr)
        rechan = AudioUtil.rechannel(reaud, channel)
        dur_aud = AudioUtil.pad_trunc(rechan, duration)
        #shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
        #sgram = AudioUtil.spectro_gram(shift_aud, n_mels, n_fft, hop_len)
        #aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        # Save the preprocessed spectrogram and the label
        torch.save((dur_aud, class_id), os.path.join(save_path, f'data_{idx}.pt'))
        
def main():
    preProcessAudio(util.from_base_path("/"), util.from_base_path("/Data/archive/"), util.from_base_path("/Data/tensors/"))

if __name__ == '__main__':
    main()