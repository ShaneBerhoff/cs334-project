import models.mobilenetv3_tl as mnv3tl
import models.inceptionv3_tl as inv3tl
import models.homebrew
from data_loader import get_loaders
import os
import argparse
import time
import util

def main(full, batch, workers, save_path):
    
    os.makedirs(save_path, exist_ok=True)
    
    # model = mnv3tl.MobileNetV3TL(full=full, save_path=save_path)
    model = inv3tl.InceptionV3TL(save_path=save_path)
    # train_dl, test_dl, train_index, test_index = get_loaders(batch_size=batch, num_workers=workers, n_mels=224, n_fft=2048, hop_len=int((24414*(2618/1000))//(224-1)-5), transform=model.transform)
    train_dl, test_dl, train_index, test_index = get_loaders(batch_size=batch, num_workers=workers, split_ratio=0.9, n_mels=224, n_fft=2048, hop_len=int((24414*(2618/1000))//(224-1)-5), transform=model.transform)
    
    with open(f'{save_path}/train_indices.txt', 'w') as f:
        for index in train_index:
            f.write(f"{index}\n")

    with open(f'{save_path}/test_indices.txt', 'w') as f:
        for index in test_index:
            f.write(f"{index}\n")
    
    print(f"Running train and predict on {model.name()} with batch size {args.batch} and {args.workers} workers")

    # train
    start = time.time()
    inv3tl.train(model, train_dl, test_dl, 50)
    print(f"Train time: {time.time() - start}")

    # predict
    start = time.time()
    inv3tl.predict(model, test_dl)
    print(f"Predict time: {time.time() - start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, help='Use full TL model')
    parser.add_argument('--batch', default=32, help='Batch size')
    parser.add_argument('--workers', default=4, help='Workers to use for data loading')
    parser.add_argument('--save', default="./Data/models/model1", help='Name of directory to save model weights')
    args = parser.parse_args()

    main(full=bool(args.full), batch=int(args.batch), workers=int(args.workers), save_path=os.path.join(util.from_base_path("/Data/models"), args.save))