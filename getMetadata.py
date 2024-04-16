import pandas as pd
from pathlib import Path
import re

def metadataCrema(directoryPath):
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
        data.append((filename, labelID))
        
    df = pd.DataFrame(data, columns=['filename', 'label'])
    
    return df

def metadataRavdess():
    return

def metadataSavee(directoryPath):
    dataPath = Path(directoryPath)
    audioFiles = dataPath.glob('*.wav')
    
    idDict = {
        "sa": 0,
        "a": 1,
        "d": 2,
        "f": 3,
        "h": 4,
        "n": 5
    }
    
    data = []
    for filePath in audioFiles:
        filename = filePath.name
        label = re.search(r"_(\D+)\d", filename).group(1)
        labelID = idDict.get(label)
        if labelID is not None:
            data.append((filename, labelID))
        
    df = pd.DataFrame(data, columns=['filename', 'label'])
    
    return df

def metadataTess():
    return
