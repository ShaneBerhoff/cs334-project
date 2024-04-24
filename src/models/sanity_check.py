import models.mobilenetv3_tl as mvn3tl
from data_loader import get_loaders
from timm.data import resolve_data_config
from torchvision.transforms import Compose, Resize, Lambda, Normalize


def repeat_channels(x):
    return x.expand(3, -1, -1)

def main():
    model = mvn3tl.MobileNetV3TL()
    config = resolve_data_config({}, model=model)
    transform = Compose([
        Resize((config['input_size'][1], config['input_size'][2])),
        Lambda(repeat_channels),  # Replicate the channel to simulate RGB
        Normalize(mean=config['mean'], std=config['std'])
    ])

    # dataloader testing
    _, test_dl, _, _ = get_loaders(batch_size=128, num_workers=0, n_mels=224, n_fft=2048, hop_len=int((24414*(2618/1000))//(224-1)-5), transform=transform)
    # first_tensor = next(iter(test_dl))
    # print(first_tensor)
    # print(first_tensor[0].shape) # should be [1, 3, 224, 224] -> 1 is batch size, 3 is channels, 224 is height, 224 is width

    for idx, (x, y) in enumerate(test_dl):
        print(x[0].shape, y[0])
    
if __name__ == '__main__':
    main()