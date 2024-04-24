from testing.model_arch import TuningAudioClassifier
import torch
from data_loader import get_test_loader

def main():
    # set up model arch
    model = TuningAudioClassifier()

    # put on gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # dataloader testing
    test_dl = get_test_loader(n_mels=224, n_fft=2048, hop_len=int((24414*(2618/1000))//(224-1)-5))
    first_tensor = next(iter(test_dl))
    print(first_tensor)
    
if __name__ == '__main__':
    main()