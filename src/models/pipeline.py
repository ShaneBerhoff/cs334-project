from models.model_params import models
from models.load_model import load_best_model
from data_loader import get_loaders
import os
import time
import util

def run_pipeline(model_info):
    save_path = os.path.join(util.from_base_path("/Data/models"), model_info["path"])
    os.makedirs(save_path, exist_ok=True)
    
    model = model_info["model"](save_path=save_path, epoch_tuning=model_info["epoch_tuning"])
    train_dl, test_dl, train_index, test_index = get_loaders(batch_size=model_info["batch"], num_workers=6, split_ratio=0.9, n_mels=model_info["n_mels"], n_fft=2048, hop_len=model_info["hop_len"], transform=model.transform)
    
    with open(f'{save_path}/train_indices.txt', 'w') as f:
        for index in train_index:
            f.write(f"{index}\n")

    with open(f'{save_path}/test_indices.txt', 'w') as f:
        for index in test_index:
            f.write(f"{index}\n")
    
    print(f"Running train and predict on {model.name()} with batch size {model_info['batch']} and 6 workers")

    # train
    start = time.time()
    model_info["package"].train(model, train_dl, test_dl, model_info["epochs"], patience=model_info["patience"])
    print(f"Train time: {time.time() - start}")

    # predict
    # load best epoch
    model, _, test_dl = load_best_model(model_info, 6)
    start = time.time()
    model_info["package"].predict(model, test_dl, final=True)
    print(f"Predict time: {time.time() - start}")


def main():
    for model in models:
        run_pipeline(models[model])


if __name__ == '__main__':
    main()