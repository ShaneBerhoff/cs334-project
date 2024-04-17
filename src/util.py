import matplotlib.pyplot as plt
import os

def show_spectrogram(sgram):
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    axs.imshow(sgram[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
    axs.set_title('Mel Spectrogram (dB)')
    axs.set_ylabel('Mel Frequency Bin')
    axs.set_xlabel('Time Window')
    plt.colorbar(axs.imshow(sgram[0].numpy(), origin='lower', aspect='auto', cmap='viridis'), ax=axs)
    plt.show()


def save_spectrogram(sgram, idx):
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    axs.imshow(sgram[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
    axs.set_title('Mel Spectrogram (dB)')
    axs.set_ylabel('Mel Frequency Bin')
    axs.set_xlabel('Time Window')
    plt.colorbar(axs.imshow(sgram[0].numpy(), origin='lower', aspect='auto', cmap='viridis'), ax=axs)
    plt.savefig(from_base_path(f'/Data/spectrogram/spectrogram_{idx}.png'))
    plt.close()


def from_base_path(path):
    filepath = os.path.dirname(__file__)
    return os.path.abspath(filepath + os.sep + os.pardir + path)
