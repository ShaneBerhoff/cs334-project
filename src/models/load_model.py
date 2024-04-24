import models.mobilenetv3_tl as mnv3tl
from data_loader import get_existing_loader
import util
import argparse
import time
import os
import re

def main(full, batch, workers, model_path, load_epoch, Train, max_epochs):
    # Fines the latest epoch file in weights and uses it
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
    
    model = mnv3tl.MobileNetV3TL(input_path=os.path.join(weights_dir, latest_file), full=full)

    train_dl, test_dl = get_existing_loader(model_path=model_path, batch_size=batch, num_workers=workers, n_mels=224, n_fft=2048, hop_len=int((24414*(2618/1000))//(224-1)-5), transform=model.transform)
    
    if Train: # train
        print("Continuing to train model weights")
        start = time.time()
        mnv3tl.train(model, train_dl, test_dl, max_epochs)
        print(f"Train time: {time.time() - start}")
    
    # predict
    print("Performing model prediction")
    start = time.time()
    mnv3tl.predict(model, test_dl)
    print(f"Predict time: {time.time() - start}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, help='Use full TL model')
    parser.add_argument('--batch', default=32, help='Batch size')
    parser.add_argument('--workers', default=4, help='Workers to use for data loading')
    parser.add_argument('--model', default=util.from_base_path("/Data/models/model1"), help='Directory name of stored model')
    parser.add_argument('--load_epoch', default=-1, help='Which stored epoch weights to use')
    parser.add_argument('--train', default=False, help='Continue training weights?')
    parser.add_argument('--max_epochs', default=20, help='Max epochs for training')
    args = parser.parse_args()
    
    main(full=bool(args.full), batch=int(args.batch), workers=int(args.workers), model_path=os.path.join(util.from_base_path("/Data/models"), args.model), load_epoch=args.load_epoch, Train=bool(args.train), max_epochs=int(args.max_epochs))