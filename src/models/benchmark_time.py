import argparse
from models.model_params import models
from models.load_model import load_best_model
import pandas as pd
import util
import time

def predict_avg(workers, k, model_info):
    """Runs predict for a model and computs the average time across k times

    Args:
        workers (int): number of subprocesses for data loading
        k (int): number of times to run predict for averaging
        model_info (dict): parameter dict for specific model

    Returns:
        int: total averaged time
        int: number of samples in test
    """
    # load best epoch
    model, _, test_dl = load_best_model(model_info, workers)
    total = 0
    for _ in range(k):
        start = time.time()
        model_info["predict"](model, test_dl, final=True)
        total += time.time() - start
    
    return total / 5, len(test_dl.dataset)

def main(k):
    """Determines the predict time for all models specified in models and outputs to /Data/benchmark_time.csv

    Args:
        k (int): number of times predict is run for averaging
    """
    df = pd.DataFrame(columns=["Model", f"Average Time (k={k})", f"Average Time per Sample (k={k})"])
    for model in models:
        print(f"Running predict on {model}")
        total, items = predict_avg(1, k, models[model])
        df.loc[len(df)] = [model, total, total / items]
        print(f"Model: {model}, Average time: {total}, Average time per sample: {total / items}")
    
    df.to_csv(util.from_base_path("/Data/benchmark_time.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=5, type=int, help='Number of times to run predict for average')
    args = parser.parse_args()
    main(k=args.k)
