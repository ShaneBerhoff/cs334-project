from src.data_loader import get_loaders
import pandas as pd

# Calc params for wanted dimensions
dimension = 224
sr = 16000
duration_ms = 2618
hop = int((sr*(duration_ms/1000))//(dimension-1))

def main():
    # Data loader
    trainDL, _ = get_loaders(n_mels=dimension, n_fft=2048, hop_len=hop)

    # Go through all data
    data = []
    for _, (inputs, _) in enumerate(trainDL):
        for spec in inputs:
            _, F, T = spec.shape
            data.append((F,T))

    df = pd.DataFrame(data, columns=['Frequency', 'Time'])
        
    print("\nFrequency Dimension Counts")
    print(df['Frequency'].value_counts())

    print("\nTime Dimension Counts")
    print(df['Time'].value_counts())    

if __name__ == '__main__':
    main()