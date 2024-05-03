from models.model_params import models
from data_loader import get_existing_loader
import util
import argparse
import time
import os
import re

def load_best_model(model_info, workers, load_epoch, options={}):
    """
    Load the best model from the weights directory alongside the correct test/train indices.

    Args:
        model: The specific model in pipeline to load.
        model_path: The path to the model directory, starting from cs334-project/Data/models.
    """
    base_path = os.path.join(util.from_base_path("/Data/models"))
    model_path = os.path.join(base_path, model_info["path"])
    weights_dir = os.path.join(model_path, "Weights")

    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"The directory {weights_dir} does not exist.")
    
    files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
    max_number = -1
    latest_file = None
    number_regex = re.compile(r'(\d+)\.pt$')
    
    for file in files:
        # Search for numbers in the filename
        match = number_regex.search(file)
        if match:
            # Convert the found number to an integer
            number = int(match.group(1))
            # Use if load_epoch selected
            if number == load_epoch:
                latest_file = file
                break
            # Update the latest file if this file has a higher number
            if number > max_number:
                max_number = number
                latest_file = file

    if latest_file is None:
        raise FileNotFoundError("No suitable .pt file found in the Weights directory.")

    print(f"Loading model with saved weights: {latest_file}")

    m = model_info["model"](input_path=os.path.join(weights_dir, latest_file), **options) # double star to unpack dictionary
    train_dl, test_dl = get_existing_loader(model_path=model_path, batch_size=model_info["batch"], num_workers=workers, n_mels=model_info["n_mels"], n_fft=2048, hop_len=model_info["hop_len"], transform=m.transform)
    
    
    return m, train_dl, test_dl


def main(workers, model_info, load_epoch, Train, max_epochs, class_acc):

    model, train_dl, test_dl = load_best_model(model_info, workers, load_epoch)
    
    if Train: # train
        print("Continuing to train model weights")
        start = time.time()
        model_info["train"](model, train_dl, test_dl, max_epochs, model_info["patience"])
        print(f"Train time: {time.time() - start}")
    
    # predict
    print(f"Performing model prediction{' with class accuracy' if class_acc else ''}")
    start = time.time()
    model_info["predict"](model, test_dl, class_acc)
    print(f"Predict time: {time.time() - start}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=6, help='Workers to use for data loading')
    parser.add_argument('--model', default="homebrew", help='Dict key of model from models')
    parser.add_argument('--load_epoch', default=-1, help='Which stored epoch weights to use')
    parser.add_argument('--train', default=False, help='Continue training weights?')
    parser.add_argument('--max_epochs', default=20, help='Max epochs for training')
    parser.add_argument('--class_acc', default=False, help='Flag for showing individual class acc during predict')
    args = parser.parse_args()
    
    main(workers=int(args.workers), model_info=models[args.model], load_epoch=args.load_epoch, Train=bool(args.train), max_epochs=int(args.max_epochs), class_acc=bool(args.class_acc))