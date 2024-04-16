import pandas as pd
from pathlib import Path

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
        data.append((filename, label, labelID))
        
    df = pd.DataFrame(data, columns=['Filename', 'Label', 'ID'])
    
    return df

def metadataRavdess(path: str):
    path = Path(path)

    idDict = {
        "04": 0, # sad
        "05": 1, # angry
        "07": 2, # disgust
        "06": 3, # fear
        "03": 4, # happy
        "01": 5  # neutral
    }

    data = []
    actors = [x for x in path.iterdir() if x.is_dir()]

    for actor in actors:
        files = actor.glob('*.wav')
        for file in files:
            filename = file.name
            labelID = idDict.get(filename.split('-')[2])
            if labelID is not None:
                data.append((filename, labelID))
    
    df = pd.DataFrame(data, columns=['filename', 'label'])

    return df

def metadataSavee():
    return

def metadataTess(path: str):
    path = Path(path)

    idDict = {
        "sad.wav": 0,
        "angry.wav": 1,
        "disgust.wav": 2,
        "fear.wav": 3,
        "happy.wav": 4,
        "neutral.wav": 5
    }

    data = []
    folders = [x for x in path.iterdir() if x.is_dir()]

    for folder in folders:
        files = folder.glob('*.wav')
        for file in files:
            filename = file.name
            labelID = idDict.get(filename.split('_')[2])
            if labelID is not None:
                data.append((filename, int(labelID)))

    df = pd.DataFrame(data, columns=['filename', 'label'])

    return df

df = metadataTess('Data/archive/Tess/')
df.to_csv('Data/TessMetadata.csv', index=False)