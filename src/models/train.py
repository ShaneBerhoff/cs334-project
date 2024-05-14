from models.model_params import models
from models.load_model import load_best_model
from data_loader import get_loaders
import os
import argparse
import time
import util

def main(batch, workers, split_ratio, model_info, class_acc):
    """Runs train and predict on a specific model storing the train/test split and weights.
    Splits are stored in /Data/models/{model_info['path']}/
    Weights are stored in /Data/models/{model_info['path']}/Weights

    Args:
        batch (int): batch size for training
        workers (int): number of subprocesses for data loading
        split_ratio (float): train test split of data
        model_info (dict): parameter dict for the specific model to load
        class_acc (bool): flag for computing individual class accuracy at runtime
    """
    save_path = os.path.join(util.from_base_path("/Data/models"), model_info["path"])
    os.makedirs(save_path, exist_ok=True)
    model = model_info["model"](save_path=save_path)

    train_dl, test_dl, train_index, test_index = get_loaders(batch_size=(model_info["batch"] if batch <= 0 else batch), num_workers=workers, split_ratio=split_ratio, n_mels=model_info["n_mels"], n_fft=2048, hop_len=model_info["hop_len"], transform=model.transform)
    
    with open(f'{save_path}/train_indices.txt', 'w') as f:
        for index in train_index:
            f.write(f"{index}\n")

    with open(f'{save_path}/test_indices.txt', 'w') as f:
        for index in test_index:
            f.write(f"{index}\n")
    
    print(f"Running train and predict on {model.name()} with batch size {model_info['batch'] if batch <= 0 else batch}, {workers} workers, and a train/test split of {split_ratio:.2f}/{1-split_ratio:.2f}")

    # train
    start = time.time()
    model_info["train"](model, train_dl, test_dl, model_info["epochs"], patience=model_info["patience"])
    print(f"Train time: {time.time() - start}")

    # predict
    # load best epoch
    model, _, test_dl = load_best_model(model_info, workers)
    start = time.time()
    model_info["predict"](model, test_dl, class_acc)
    print(f"Predict time: {time.time() - start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=-1, type=int, help='Batch size')
    parser.add_argument('--workers', default=6, type=int, help='Workers to use for data loading')
    parser.add_argument('--split', default=0.9, type=float, help='Percent of data for train')
    parser.add_argument('--model', default="homebrew", help=f"Options: {[model for model in models]}")
    parser.add_argument('--class_acc', action='store_true', help='Flag for showing individual class accuracy during predict')
    args = parser.parse_args()

    main(batch=int(args.batch), workers=int(args.workers), split_ratio=args.split, model_info=models[args.model], class_acc=args.class_acc)