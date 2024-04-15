import pandas as pd
from pathlib import Path

def metadataDF(directoryPath):
    dataPath = Path(directoryPath)
    audioFiles = dataPath.glob('*.wav')
    
    idDict = {
        "SAD": 0,
        "ANG": 1,
        "DIS": 2,
        "FEA": 3,
        "HAP": 4,
        "NEU": 5
    }
    data = []
    for filePath in audioFiles:
        filename = filePath.name
        label = filename.split('_')[2]
        labelID = idDict[label]
        data.append((filename, label, labelID))
        
    df = pd.DataFrame(data, columns=['Filename', 'Label', 'ID'])
    
    return df

df = metadataDF('Data/archive/Crema')
df.to_csv('Data/CremaMetadata.csv', index=False)