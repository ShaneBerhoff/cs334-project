from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import AudioUtil
import pandas as pd
import os
import metadata
import util
import torch

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, data_path, shift_pct=0.3):
        self.data_path = str(data_path)
        self.shift_pct = shift_pct
        self.len = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
                        
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return self.len    
        
    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # tensorfile
        tensorFile = torch.load(os.path.join(self.data_path, f'data_{idx}.pt'))
        # Unpack
        audio, classID = tensorFile

        # Load audio file
        #aud = AudioUtil.load_audio(audio_file)
        #reaud = AudioUtil.resample(aud, self.sr)
        #rechan = AudioUtil.rechannel(reaud, self.channel)
        #dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        
        # Shift audio
        shift_aud = AudioUtil.time_shift(audio, self.shift_pct)
        # Create spectrogram
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=256)
        # Augment spectrogram
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, classID


def get_loaders(batch_size=32, split_ratio=0.8, num_workers=4):
    data_path = util.from_base_path("/Data/tensors/")
    myds = SoundDS(data_path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * split_ratio)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl


def get_subsample_loaders(batch_size=32, split_ratio=0.8, num_workers=4):
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
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl

# Check for working
if __name__ == '__main__':
    data_path = "/"
    df = metadata.Metadata(util.from_base_path("/Data/archive/")).getMetadata()
    print("Label counts (0 = sad, 1 = angry, 2 = disgust, 3 = fear, 4 = happy, 5 = neutral):\n",df["label"].value_counts())
    myds = SoundDS(df, data_path)

    for i in range(5):
        util.save_spectrogram(myds[i][0], i)