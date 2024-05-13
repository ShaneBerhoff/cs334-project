from models.model_params import models
from data_loader import get_loaders
import matplotlib.pyplot as plt

def main():
    """
    Generates spectrograms for all defined models.
    Used to check dimension of data fed into each model.
    Used to verify correct spectrogram generation.
    """
    for m in models:
        model_info = models[m]
        model = model_info["model"]()
        train_dl, _, _, _ = get_loaders(batch_size=1, num_workers=6, n_mels=model_info["n_mels"], n_fft=2048, hop_len=model_info["hop_len"], transform=model.transform)
        
        for sgram, _ in train_dl:
            print(f"Item in {model.name()} of shape: {sgram.shape}")
            # Check if sgram is a single or multi-channel image and adjust accordingly
            if sgram.shape[1] == 1:  # Single channel
                image_data = sgram[0].numpy().squeeze()
                cmap = 'viridis'  # Suitable for single channel data
            # Adjust the tensor for display
            elif sgram.shape[1] == 3:  # Confirming that it's an RGB image
                image_data = sgram[0].permute(1, 2, 0).numpy()  # Remove batch dimension and permute to [H, W, C]
                cmap = None

            # Setting up the plot
            fig, axs = plt.subplots(1, 1, figsize=(10, 4))
            im = axs.imshow(image_data, origin='lower', aspect='auto', cmap=cmap)
            axs.set_title('Mel Spectrogram (dB)')
            axs.set_ylabel('Mel Frequency Bin')
            axs.set_xlabel('Time Window')
            plt.colorbar(im, ax=axs)
            
            plt.savefig(f'Data/Spectrogram/{model.name()}.png')
            plt.close()
            break
        
if __name__ == '__main__':
    main()