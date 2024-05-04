from models.pipeline import models
from models.load_model import load_best_model
import pandas as pd
import os
import util
import time

def predict_avg(batch, workers, model_info):
    save_path = os.path.join(util.from_base_path("/Data/models"), model_info["path"])
    os.makedirs(save_path, exist_ok=True)

    # load best epoch
    model, _, test_dl = load_best_model(model_info, workers)
    total = 0
    for i in range(5):
        start = time.time()
        model_info["package"].predict(model, test_dl, final=True)
        total += time.time() - start
    
    return total / 5

def main():
    df = pd.DataFrame(columns=["Model", "Average Time (k=5)", "Average Time per Sample (k=5)"])
    for model in models:
        print(f"Running predict on {model}")
        total = predict_avg(1, 1, models[model])
        df.loc[len(df)] = [model, total, total / 1132]
        print(f"Model: {model}, Average time: {total}, Average time per sample: {total / 1132}")
    
    df.to_csv(util.from_base_path("/Data/benchmark_time.csv"), index=False)

if __name__ == '__main__':
    main()
