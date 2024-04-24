from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import AudioUtil
import os
import metadata
import util
import torch
from torch.utils.data import Subset

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, data_path, shift_pct, n_mels, n_fft, hop_len, max_mask_pct, n_masks, transform):
        self.data_path = str(data_path)
        self.shift_pct = shift_pct
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.max_mask_pct = max_mask_pct
        self.n_masks = n_masks
        self.len = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
        self.transform = transform
        self.val_indices = None
    
    def __len__(self):
        return self.len    

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
        sgram = AudioUtil.spectro_gram(shift_aud, self.n_mels, self.n_fft, self.hop_len)

        # Augment spectrogram
        if idx in self.val_indices:
            aug_sgram = sgram
        else:
            aug_sgram = AudioUtil.spectro_augment(sgram, self.max_mask_pct, n_freq_masks=self.n_masks, n_time_masks=self.n_masks)

        # Convert spectrogram to an image and apply final transformations
        if self.transform:
            aug_sgram = self.transform(aug_sgram)

        return aug_sgram, classID

    def set_val_indices(self, val_indices):
        self.val_indices = val_indices


def get_loaders(batch_size=32, split_ratio=0.8, num_workers=4, shift_pct=0.3, n_mels=64, n_fft=1024, hop_len=256, max_mask_pct=0.1, n_masks=2, transform=None):
    data_path = util.from_base_path("/Data/tensors/")
    myds = SoundDS(data_path, shift_pct, n_mels, n_fft, hop_len, max_mask_pct, n_masks, transform)

    num_items = len(myds)
    num_train = round(num_items * split_ratio)
    num_val = num_items - num_train
    train_indices, val_indices = random_split(range(num_items), [num_train, num_val])
    myds.set_val_indices(val_indices)

    # Create separate datasets for training and validation
    train_ds = Subset(myds, train_indices)
    val_ds = Subset(myds, val_indices)

    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, train_indices, val_indices


def get_existing_loader(model_path="/Data/models/model1", batch_size=32, num_workers=4, shift_pct=0.3, n_mels=64, n_fft=1024, hop_len=256, max_mask_pct=0.1, n_masks=2, transform=None):
    data_path = util.from_base_path("/Data/tensors/")
    myds = SoundDS(data_path, shift_pct, n_mels, n_fft, hop_len, max_mask_pct, n_masks, transform)

    # Get train subset
    with open(os.path.join(model_path, "train_indices.txt"), 'r') as file:
        indices1 = file.readlines()
        # Convert each line to an integer
    indices1 = [int(idx.strip()) for idx in indices1]
    
    train_ds = Subset(myds, indices1)
    
    # Get test subset
    with open(os.path.join(model_path, "test_indices.txt"), 'r') as file:
        indices = file.readlines()
        # Convert each line to an integer
    indices = [int(idx.strip()) for idx in indices]
    
    test_ds = Subset(myds, indices)
    
    # Create loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, test_dl


# Check for working
#if __name__ == '__main__':
#    data_path = "/"
#    df = metadata.Metadata(util.from_base_path("/Data/archive/")).getMetadata()
#    print("Label counts (0 = sad, 1 = angry, 2 = disgust, 3 = fear, 4 = happy, 5 = neutral):\n",df["label"].value_counts())
#    myds = SoundDS(df, data_path)
#
#    for i in range(5):
#        util.save_spectrogram(myds[i][0], i)