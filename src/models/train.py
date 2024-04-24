from models.pipeline import models
from data_loader import get_loaders
import os
import argparse
import time
import util

def main(full, batch, workers, save_path, model_info):
    
    os.makedirs(save_path, exist_ok=True)
    
    # model = mnv3tl.MobileNetV3TL(full=full, save_path=save_path)
    # model = dn121tl.DenseNet121TL(save_path=save_path)
    model = model_info["model"](full=full, save_path=save_path)

    # train_dl, test_dl, train_index, test_index = get_loaders(batch_size=batch, num_workers=workers, n_mels=224, n_fft=2048, hop_len=int((24414*(2618/1000))//(224-1)-5), transform=model.transform)
    train_dl, test_dl, train_index, test_index = get_loaders(batch_size=(model_info["batch"] if batch <= 0 else batch), num_workers=workers, split_ratio=0.9, n_mels=model_info["n_mels"], n_fft=2048, hop_len=model_info["hop_len"], transform=model.transform)
    
    with open(f'{save_path}/train_indices.txt', 'w') as f:
        for index in train_index:
            f.write(f"{index}\n")

    with open(f'{save_path}/test_indices.txt', 'w') as f:
        for index in test_index:
            f.write(f"{index}\n")
    
    print(f"Running train and predict on {model.name()} with batch size {model_info['batch'] if batch <= 0 else batch} and {workers} workers")

    # train
    start = time.time()
    model_info["package"].train(model, train_dl, test_dl, model_info["epochs"], patience=model_info["patience"])
    print(f"Train time: {time.time() - start}")

    # predict
    start = time.time()
    model_info["package"].predict(model, test_dl)
    print(f"Predict time: {time.time() - start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, help='Use full TL model')
    parser.add_argument('--batch', default=-1, help='Batch size')
    parser.add_argument('--workers', default=6, help='Workers to use for data loading')
    parser.add_argument('--save', default="./Data/models/model1", help='Name of directory to save model weights')
    parser.add_argument('--model', default="mnv3tl", help=f"Options: {[model for model in models]}")

    args = parser.parse_args()

    main(full=bool(args.full), batch=int(args.batch), workers=int(args.workers), save_path=os.path.join(util.from_base_path("/Data/models"), args.save), model_info=models[args.model])