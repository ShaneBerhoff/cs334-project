import pandas as pd
from pathlib import Path
import re

class Metadata():
    path = None
    metadata = None
    
    def __init__(self, directoryPath: str) -> None:
        self.path = Path(directoryPath)

    def getMetadata(self) -> pd.DataFrame:
        """Compiles metatdata informaion from all datasets to one place.

        Returns:
            pd.DataFrame: filepath and rescaled label for each sample
        """
        folders = [x for x in self.path.iterdir() if x.is_dir()]
        data = pd.DataFrame()
        for folder in folders:
            if hasattr(self, folder.name):
                method = getattr(self, folder.name)
                df = method(folder)
                data = pd.concat([data, df], ignore_index=True, sort=False)
            else:
                print(f"No method found for {folder.name}")
        self.metadata = data
        return self.metadata
    
    @staticmethod
    def Crema(directoryPath: str):
        path = Path(directoryPath)
        audioFiles = path.glob('*.wav')
        
        idDict = {
            "SAD": 0, # sad
            "ANG": 1, # angry
            "DIS": 2, # disgust
            "FEA": 3, # fear 
            "HAP": 4, # happy
            "NEU": 5  # neutral
        }
        data = []
        for filePath in audioFiles:
            filename = filePath.name
            label = filename.split('_')[2]
            labelID = idDict[label]
            data.append((filePath, labelID))
            
        df = pd.DataFrame(data, columns=['filepath', 'label'])
        
        return df

    @staticmethod
    def Ravdess(directoryPath: str):
        path = Path(directoryPath)
        path = next((x for x in path.iterdir() if x.is_dir()))
        
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
                    data.append((file, labelID))
        
        df = pd.DataFrame(data, columns=['filepath', 'label'])

        return df

    @staticmethod
    def Savee(directoryPath: str):
        path = Path(directoryPath)
        audioFiles = path.glob('*.wav')
        
        idDict = {
            "sa": 0, # sad
            "a": 1,  # angry
            "d": 2,  # disgust
            "f": 3,  # fear
            "h": 4,  # happy
            "n": 5   # neutral
        }
        
        data = []
        for filePath in audioFiles:
            filename = filePath.name
            match = re.search(r"_(\D+)\d", filename)
            if match:
                label = match.group(1)
                labelID = idDict.get(label)
                if labelID is not None:
                    data.append((filePath, labelID))

        df = pd.DataFrame(data, columns=['filepath', 'label'])
        
        return df

    @staticmethod
    def Tess(directoryPath: str):
        path = Path(directoryPath)

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
                    data.append((file, int(labelID)))

        df = pd.DataFrame(data, columns=['filepath', 'label'])

        return df

# Usage
# data = Metadata(util.from_base_path('/Data/archive/'))
# df = data.getMetadata()