import models.mobilenetv3_tl as mnv3tl
import models.inceptionv3_tl as inv3tl
import models.densenet121_tl as dn121tl
import models.efficientnetv2b0_tl as env2b0tl
from data_loader import get_loaders
import os
import time
import util

def run_pipeline(model_info):
    save_path = os.path.join(util.from_base_path("/Data/models"), model_info["path"])
    
    os.makedirs(save_path, exist_ok=True)
    
    
    model = model_info["model"](save_path=save_path)
    # train_dl, test_dl, train_index, test_index = get_loaders(batch_size=batch, num_workers=workers, n_mels=224, n_fft=2048, hop_len=int((24414*(2618/1000))//(224-1)-5), transform=model.transform)
    train_dl, test_dl, train_index, test_index = get_loaders(batch_size=model_info["batch"], num_workers=6, split_ratio=0.9, n_mels=224, n_fft=2048, hop_len=int((24414*(2618/1000))//(224-1)-5), transform=model.transform)
    
    with open(f'{save_path}/train_indices.txt', 'w') as f:
        for index in train_index:
            f.write(f"{index}\n")

    with open(f'{save_path}/test_indices.txt', 'w') as f:
        for index in test_index:
            f.write(f"{index}\n")
    
    print(f"Running train and predict on {model.name()} with batch size {model_info['batch']} and 6 workers")

    # train
    start = time.time()
    model_info["package"].train(model, train_dl, test_dl, 50)
    print(f"Train time: {time.time() - start}")

    # predict
    start = time.time()
    model_info["package"].predict(model, test_dl)
    print(f"Predict time: {time.time() - start}")


def main():
    models = {
        "mnv3tl": {
            "package": mnv3tl,
            "model": mnv3tl.MobileNetV3TL,
            "train": mnv3tl.train,
            "predict": mnv3tl.predict,
            "path": "mnv3tl-pipeline",
            "batch": 64
        },
        "env2b0tl": {
            "package": env2b0tl,
            "model": env2b0tl.EfficientNetV2B0TL,
            "train": env2b0tl.train,
            "predict": env2b0tl.predict,
            "path": "env2b0tl-pipeline",
            "batch": 64
        },
        "inv3tl": {
            "package": inv3tl,
            "model": inv3tl.InceptionV3TL,
            "train": inv3tl.train,
            "predict": inv3tl.predict,
            "path": "inv3tl-pipeline",
            "batch": 32
        },
        "dn121tl": {
            "package": dn121tl,
            "model": dn121tl.DenseNet121TL,
            "train": dn121tl.train,
            "predict": dn121tl.predict,
            "path": "dn121tl-pipeline",
            "batch": 64
        }
    }

    for model in models:
        run_pipeline(models[model])


if __name__ == '__main__':
    main()