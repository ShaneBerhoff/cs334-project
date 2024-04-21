from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import AudioUtil
import pandas as pd
import os
import metadata
import util

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path, duration=2618, sr=16000, channel=2, shift_pct=0.3):
        self.df = df
        self.data_path = str(data_path)
        self.duration = duration
        self.sr = sr
        self.channel = channel
        self.shift_pct = shift_pct
                        
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    
        
    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = os.path.join(self.data_path, self.df.loc[idx, 'filepath'])
        # Get the Class ID
        class_id = self.df.loc[idx, 'label']

        # Load audio file
        aud = AudioUtil.load_audio(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        
        # Shift audio
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        # Create spectrogram
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=256)
        # Augment spectrogram
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id


def get_loaders(batch_size=32, split_ratio=0.8, num_workers=4):
    data_path = util.from_base_path("/")
    df = metadata.Metadata(util.from_base_path("/Data/archive/")).getMetadata()
    myds = SoundDS(df, data_path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * split_ratio)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl


def get_subsample_loaders(batch_size=16, split_ratio=0.8, num_workers=0):
    data_path = util.from_base_path("/")
    df = metadata.Metadata(util.from_base_path("/Data/archive/")).getMetadata()

    # select class 0 and 1, randomly select 10% of samples
    df_sad = df[df["label"].isin([1])] # sad
    df_happy = df[df["label"].isin([4])] # happy
    df_sub = pd.concat([df_sad.sample(frac=0.1), df_happy.sample(frac=0.1)], ignore_index=True)
    subds = SoundDS(df_sub, data_path)

    # Random split of 80:20 between training and validation
    num_items = len(subds)
    num_train = round(num_items * split_ratio)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(subds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl

# Check for working
if __name__ == '__main__':
    data_path = "/"
    df = metadata.Metadata(util.from_base_path("/Data/archive/")).getMetadata()
    print("Label counts (0 = sad, 1 = angry, 2 = disgust, 3 = fear, 4 = happy, 5 = neutral):\n",df["label"].value_counts())
    myds = SoundDS(df, data_path)

    for i in range(5):
        util.save_spectrogram(myds[i][0], i)