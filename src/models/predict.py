from models.pipeline import models
from models.load_model import load_best_model
import os
import argparse
import time
import util

def main(batch, workers, model_info):
    save_path = os.path.join(util.from_base_path("/Data/models"), model_info["path"])
    os.makedirs(save_path, exist_ok=True)

    # load best epoch
    model, _, test_dl = load_best_model(model_info, workers)
    print(f"Running predict on {model.name()} with batch size {model_info['batch'] if batch <= 0 else batch} and {workers} workers")

    # predict
    start = time.time()
    model_info["package"].predict(model, test_dl)
    print(f"Predict time: {time.time() - start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=-1, help='Batch size')
    parser.add_argument('--workers', default=6, help='Workers to use for data loading')
    parser.add_argument('--model', default="mnv3tl", help=f"Options: {[model for model in models]}")

    args = parser.parse_args()

    main(batch=int(args.batch), workers=int(args.workers), model_info=models[args.model])