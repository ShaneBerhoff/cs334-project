import pandas as pd
from pathlib import Path
import torchaudio

def Crema(directoryPath):
    dataPath = Path(directoryPath)
    audioFiles = dataPath.glob('*.wav')
    
    data = []
    for filePath in audioFiles:
        waveform, sr = torchaudio.load(filePath)
        length = waveform.shape[1] / sr
        data.append((sr, length))
        
    df = pd.DataFrame(data, columns=['sampleRate', 'length'])
    
    return df

def Ravdess(path: str):
    path = Path(path)

    data = []
    actors = [x for x in path.iterdir() if x.is_dir()]

    for actor in actors:
        files = actor.glob('*.wav')
        for file in files:
            waveform, sr = torchaudio.load(file)
            length = waveform.shape[1] / sr
            data.append((sr, length))
    
    df = pd.DataFrame(data, columns=['sampleRate', 'length'])

    return df

def Savee(directoryPath):
    dataPath = Path(directoryPath)
    audioFiles = dataPath.glob('*.wav')
    
    data = []
    for filePath in audioFiles:
        waveform, sr = torchaudio.load(filePath)
        length = waveform.shape[1] / sr
        data.append((sr, length))
        
    df = pd.DataFrame(data, columns=['sampleRate', 'length'])
    
    return df

def Tess(path: str):
    path = Path(path)

    data = []
    folders = [x for x in path.iterdir() if x.is_dir()]

    for folder in folders:
        files = folder.glob('*.wav')
        for file in files:
            waveform, sr = torchaudio.load(file)
            length = waveform.shape[1] / sr
            data.append((sr, length))

    df = pd.DataFrame(data, columns=['sampleRate', 'length'])

    return df

dfCrema = Crema('Data/archive/Crema/')
dfRavdess = Ravdess('Data/archive/Ravdess/audio_speech_actors_01-24/')
dfSavee = Savee('Data/archive/Savee/')
dfTess = Tess('Data/archive/Tess/')

df = pd.concat([dfCrema, dfRavdess, dfSavee, dfTess], ignore_index=True, sort=False)

print("\nSample Rate Counts")
print(df['sampleRate'].value_counts())

print("\nAverage Length of Clip (s)")
print(df['length'].mean())