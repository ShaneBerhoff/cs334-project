import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_arch import TuningAudioClassifier
from predict import predict
from training import training
import torch
from data_loader import get_subsample_loaders
import time

# set up model arch
model = TuningAudioClassifier()

# put on gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

start = time.time()
# set up dataloader
trainDL, testDL = get_subsample_loaders()
print(f"Data loader time: {time.time() - start}")

# train model
start = time.time()
training(model, trainDL, 1)
print(f"Train time: {time.time() - start}")

# predictions
start = time.time()
predict(model, testDL)
print(f"Predict time: {time.time() - start}")