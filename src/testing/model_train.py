import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_arch import TuningAudioClassifier
from predict import predict
from training import training
import torch
from data_loader import get_loaders
import time

def main():
    # set up model arch
    model = TuningAudioClassifier()

    # put on gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    start = time.time()
    # set up dataloader
    trainDL, testDL = get_loaders(n_mels=224, n_fft=2048, hop_len=int((24414*(2618/1000))//(224-1)-5))
    print(f"Data loader time: {time.time() - start}")

    # train model
    start = time.time()
    training(model, trainDL, 20, device)
    print(f"Train time: {time.time() - start}")

    # predictions
    start = time.time()
    predict(model, testDL, device)
    print(f"Predict time: {time.time() - start}")
    
if __name__ == '__main__':
    main()