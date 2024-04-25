import models.mobilenetv3_tl as mnv3tl
import models.efficientnetv2b0_tl as env2b0tl
import models.efficientnetv2b1_tl as env2b1tl
import models.inceptionv3_tl as inv3tl
import models.densenet121_tl as dn121tl
import models.homebrew as homebrew
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
    model_info["package"].predict(model, test_dl)
    print(f"Predict time: {time.time() - start}")

models = {
        # "mnv3tl": {
        #     "package": mnv3tl,
        #     "model": mnv3tl.MobileNetV3TL,
        #     "train": mnv3tl.train,
        #     "predict": mnv3tl.predict,
        #     "path": "mnv3tl-pipeline3",
        #     "batch": 64,
        #     "epochs": 15,
        #     "epoch_tuning": False,
        #     "patience": 5,
        #     "n_mels": 224, # required dimension of 224x224
        #     "hop_len": 281 # from magic formula ((24414*(2618/1000))//(224-1)-5)
        # },
        # "env2b0tl": {
        #     "package": env2b0tl,
        #     "model": env2b0tl.EfficientNetV2B0TL,
        #     "train": env2b0tl.train,
        #     "predict": env2b0tl.predict,
        #     "path": "env2b0tl-pipeline3",
        #     "batch": 64,
        #     "epochs": 18,
        #     "epoch_tuning": False,
        #     "patience": 5,
        #     "n_mels": 192, # required dimension of 192x192
        #     "hop_len": 328 # from magic formula ((24414*(2618/1000))//(192-1)-6)
        # },
        # "env2b1tl": {
        #     "package": env2b1tl,
        #     "model": env2b1tl.EfficientNetV2B1TL,
        #     "train": env2b1tl.train,
        #     "predict": env2b1tl.predict,
        #     "path": "env2b1tl-pipeline3",
        #     "batch": 64,
        #     "epochs": 11,
        #     "epoch_tuning": False,
        #     "patience": 5,
        #     "n_mels": 192, # required dimension of 192x192
        #     "hop_len": 328 # from magic formula ((24414*(2618/1000))//(192-1)-6)
        # },
        "homebrew": {
            "package": homebrew,
            "model": homebrew.Homebrew,
            "train": homebrew.train,
            "predict": homebrew.predict,
            "path": "homebrew-pipeline3",
            "batch": 64,
            "epochs": 2,
            "epoch_tuning": False,
            "patience": 5,
            "n_mels": 128,
            "hop_len": 512
        },
        # "inv3tl": {
        #     "package": inv3tl,
        #     "model": inv3tl.InceptionV3TL,
        #     "train": inv3tl.train,
        #     "predict": inv3tl.predict,
        #     "path": "inv3tl-pipeline3",
        #     "batch": 32,
        #     "epochs": 15,
        #     "patience": 5,
        #     "n_mels": 299, # required dimension of 299x299
        #     "hop_len": 211 # from magic formula ((24414*(2618/1000))//(299-1)-3)
        # },
        # "dn121tl": {
        #     "package": dn121tl,
        #     "model": dn121tl.DenseNet121TL,
        #     "train": dn121tl.train,
        #     "predict": dn121tl.predict,
        #     "path": "dn121tl",
        #     "batch": 64,
        #     "epochs": 30,
        #     "patience": 8,
        #     "n_mels": 224, #required dimension of 224x224
        #     "hop_len": 281 # from magic formula ((24414*(2618/1000))//(224-1)-5)
        # }
    }


def main():
    for model in models:
        run_pipeline(models[model])


if __name__ == '__main__':
    main()