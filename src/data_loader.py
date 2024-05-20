from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import AudioUtil
import os
import util
import torch
from torch.utils.data import Subset


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
        
        # Augment audio only if train data
        if idx in self.val_indices:
            # Create spectrogram
            aug_sgram = AudioUtil.spectro_gram(audio, self.n_mels, self.n_fft, self.hop_len)
        else:
            # Shift audio & create spectrogram
            shift_aud = AudioUtil.time_shift(audio, self.shift_pct)
            sgram = AudioUtil.spectro_gram(shift_aud, self.n_mels, self.n_fft, self.hop_len)
            aug_sgram = AudioUtil.spectro_augment(sgram, self.max_mask_pct, n_freq_masks=self.n_masks, n_time_masks=self.n_masks)

        # Apply final normalized transformation
        if self.transform:
            aug_sgram = self.transform(aug_sgram)

        return aug_sgram, classID

    def set_val_indices(self, val_indices):
        self.val_indices = val_indices


def get_loaders(batch_size=32, split_ratio=0.8, num_workers=4, shift_pct=0.3, n_mels=64, n_fft=1024, hop_len=256, max_mask_pct=0.1, n_masks=2, transform=None):
    """Produces train and test dataloaders.
    Uses data stored in /Data/tensors.

    Args:
        batch_size (int, optional): size of batches. Defaults to 32.
        split_ratio (float, optional): percent of data that goes to training. Defaults to 0.8.
        num_workers (int, optional): number of subprocesses for dataloading. Defaults to 4.
        shift_pct (float, optional): max audio shift percent. Defaults to 0.3.
        n_mels (int, optional): number of mels for spectrogram. Defaults to 64.
        n_fft (int, optional): fft for spectrogram. Defaults to 1024.
        hop_len (int, optional): hop len for spectrogram. Defaults to 256.
        max_mask_pct (float, optional): max percent of spectrogram to mask. Defaults to 0.1.
        n_masks (int, optional): number of masks per domain. Defaults to 2.
        transform (_type_, optional): data transformation for pretrained normalizion. Defaults to None.

    Returns:
        train_dataloader, test_dataloader, train_indices, test_indices
    """
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


def get_existing_loader(model_path="/Data/models/test-homebrew", batch_size=32, num_workers=4, shift_pct=0.3, n_mels=64, n_fft=1024, hop_len=256, max_mask_pct=0.1, n_masks=2, transform=None):
    """Produces train and test dataloaders based on previously defined model data splits.
    Uses data stored in /Data/tensors.

    Args:
        model_path (str): path of stored model info. 
        batch_size (int, optional): size of batches. Defaults to 32.
        num_workers (int, optional): number of subprocesses for dataloading. Defaults to 4.
        shift_pct (float, optional): max audio shift percent. Defaults to 0.3.
        n_mels (int, optional): number of mels for spectrogram. Defaults to 64.
        n_fft (int, optional): fft for spectrogram. Defaults to 1024.
        hop_len (int, optional): hop len for spectrogram. Defaults to 256.
        max_mask_pct (float, optional): max percent of spectrogram to mask. Defaults to 0.1.
        n_masks (int, optional): number of masks per domain. Defaults to 2.
        transform (_type_, optional): data transformation for pretrained normalizion. Defaults to None.

    Returns:
        train_dataloader, test_dataloader
    """
    data_path = util.from_base_path("/Data/tensors/")
    myds = SoundDS(data_path, shift_pct, n_mels, n_fft, hop_len, max_mask_pct, n_masks, transform)

    # Get indices
    with open(os.path.join(model_path, "train_indices.txt"), 'r') as file:
        train_indices = [int(idx.strip()) for idx in file.readlines()]
    with open(os.path.join(model_path, "test_indices.txt"), 'r') as file:
        val_indices = [int(idx.strip()) for idx in file.readlines()]
    
    myds.set_val_indices(val_indices)

    # Create separate datasets for training and validation
    train_ds = Subset(myds, train_indices)
    test_ds = Subset(myds, val_indices)
    
    # Create loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, test_dl
