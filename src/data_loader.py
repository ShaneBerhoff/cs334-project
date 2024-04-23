from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import AudioUtil
import os
import metadata
import util
import torch
import torchvision.transforms as transforms


def expand_dim(tensor):
    return tensor.expand(3, -1, -1)

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, data_path, shift_pct, n_mels, n_fft, hop_len, max_mask_pct, n_masks, model=None): # model for eventual modularization?
        self.data_path = str(data_path)
        self.shift_pct = shift_pct
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.max_mask_pct = max_mask_pct
        self.n_masks = n_masks
        self.len = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
        self.transform = transforms.Compose([
                transforms.Lambda(expand_dim),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
                        
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
        sgram = AudioUtil.spectro_gram(shift_aud, self.n_mels, self.n_fft, self.hop_len)
        # Augment spectrogram
        aug_sgram = AudioUtil.spectro_augment(sgram, self.max_mask_pct, n_freq_masks=self.n_masks, n_time_masks=self.n_masks)

        # Transform for model
        transform_sgram = self.transform(aug_sgram)

        return transform_sgram, classID


def get_loaders(batch_size=32, split_ratio=0.8, num_workers=4, shift_pct=0.3, n_mels=64, n_fft=1024, hop_len=256, max_mask_pct=0.1, n_masks=2):
    data_path = util.from_base_path("/Data/tensors/")
    myds = SoundDS(data_path, shift_pct, n_mels, n_fft, hop_len, max_mask_pct, n_masks, model='mobilenetv3')

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * split_ratio)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl


def get_test_loader(n_mels, n_fft, hop_len):
    data_path = util.from_base_path("/Data/tensors/")
    myds = SoundDS(data_path, shift_pct=0.3, n_mels=n_mels, n_fft=n_fft, hop_len=hop_len, max_mask_pct=0.1, n_masks=2, model='mobilenetv3')
    return DataLoader(myds, batch_size=1, shuffle=False)


# Check for working
if __name__ == '__main__':
    data_path = "/"
    df = metadata.Metadata(util.from_base_path("/Data/archive/")).getMetadata()
    print("Label counts (0 = sad, 1 = angry, 2 = disgust, 3 = fear, 4 = happy, 5 = neutral):\n",df["label"].value_counts())
    myds = SoundDS(df, data_path)

    for i in range(5):
        util.save_spectrogram(myds[i][0], i)