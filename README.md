# Efficient Speech Emotion Classification

## Overview
This project focuses on identifying human emotions from speech audio alone, without relying on text transcription. This was done as a final project for CS 334 (Machine Learning) and utilizes various datasets to enhance the robustness of the emotion classification model. For detailed information on the project goals and results, please refer to the [Final Report](/Final_Report.pdf).

## Installation
Install all necessary dependencies from the provided `requirements.txt` file to ensure the software runs correctly.

## Data Preparation
The [data](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en) for this project is sourced from Kaggle and includes the following datasets: Crema, Ravdess, Savee, and Tess. Ensure that these datasets are stored at `./Data/archive/`. Use the script `./src/data_stats.py` to analyze the audio data's sample rate distribution, average length, and channel count. Based on this analysis, select appropriate values of each for preprocessing.

### Previous Settings
For this project, the following preprocessing settings were selected based on the initial analysis:
- **Sample Rate**: 24414 Hz
- **Duration**: 2618 ms
- **Channels**: 1

This information provides a reference point for setting up the data preprocessing.

### Preprocessing
Run `./src/preprocessing.py` with the chosen values to standardize and convert all dataset `.wav` files into PyTorch `.pt` files. The previous settings are set as default values. The processed files will be saved to `./Data/tensors` by default.

## Model Setup
The project includes various model architectures under `./src/models/arch`. Each model file contains custom class definitions, including methods for training, prediction, and saving models.

### Adding New Models
To integrate a new model:
1. Create its architecture file in the `./src/models/arch` directory.
2. Define the model in `./src/models/model_params.py` adding it to the `models` dictionary using the following format:

```python
import models.arch.fileName as package

"name_key": {
    "package": package,
    "model": package.ClassName,
    "train": package.train,
    "predict": package.predict,
    "path": "save-path-name",  # Saves to ./Data/models/save-path-name
    "batch": batch_size_for_training,
    "epochs": max_epochs_to_run,
    "epoch_tuning": True/False,
    "patience": patience_for_early_stopping,
    "n_mels": number_of_mels_for_spectrogram,
    "hop_len": hop_length_for_spectrogram
}
```
For more details on adjusting `n_mels` and `hop_len` for different models, see the calculations provided in section 3.4.2 of the [Final Report](/Final_Report.pdf).

### Checking Samples and Dimension
Use `./src/models/spectrogram_check.py` to output what the dimensions of spectrogram samples will be at train-time for each model. These should align with required input dimensions for any transfer learning models. It also saves a visual representation of a sample to `./Data/Spectrogram/{model.name()}.png`.

### Training Models
Run `./src/models/train.py` from the command line to train any model listed in the `models` dictionary. The script manages data splits for reproducibility and saves model weights upon performance improvements. A prediction analysis is performed at the end of training to evaluate the model.

For batch processing of all models, use `./src/models/pipeline.py`, which will train and save all defined models.

#### Previous Parameters
The optimal batch sizing and number of workers to achieve minimum train time per epoch can vary based on the hardware used. The goal is to maximize GPU utilization while making efficient use of CPU resources. Below are the configurations for the two devices used in the project along with their optimal training parameters:

- **Acer Swift X 2021**:
  - **CPU**: AMD Ryzen 7 5800U @ 1.9 GHz
  - **GPU**: Nvidia GeForce RTX 3050 Ti @ 40W
  - **RAM**: 16 GB
  - **Optimal Batch Size**: 64
  - **Optimal Workers**: 6
- **MacBook Pro M1 Pro**:
  - **CPU**: 10-core CPU
  - **GPU**: 16-core GPU
  - **RAM**: 32 GB
  - **Optimal Batch Size**: 64
  - **Optimal Workers**: 1

### Model Prediction and Management
Use `./src/models/load_model.py` to load a previously trained model for prediction, with the ability to continue training. Specify the model using its `name_key` from the `models` dictionary.

For performance benchmarking across models, `./src/models/benchmark_time.py` measures and logs the prediction time complexity, with results saved to `./Data/benchmark_time.csv`.

## Additional

Other files that do not need to be directly used due to abstraction or a testing use case.

### `./testing/testDimension.py`
Used to test the output dimension of spectrograms based on different input values.

### `./testing/testpre.py`
Used to test all parts of the AudioUtil class from preprocessing for processing samples.

### `./src/metadata.py`
Collects class information for samples from Crema, Ravdess, Savee, and Tess only selecting classes that are common between all four datasets. Standardizes them to a class scheme of:
```python
IDMAP = {
        0: "Sad",
        1: "Angry",
        2: "Disgust",
        3: "Fear",
        4: "Happy",
        5: "Neutral"
}
```
Produces a Pandas DataFrame where each row contains the path to a sample and its standardized class. This is used by `./src/preprocessing.py` in conversion to tensors.

### `./src/data_loader.py`
- `get_loaders()`: produces train and validation dataloaders with a random split on data stored in `./Data/tensors`.
- `get_existing_loader()`: uses save information of a model to reproduce train and validation loaders with the same samples that the model was trained on. 
Used throughout training and prediction related files.

### `./src/util.py`
Holds the `IDMAP` definitions, helpers to generate spectrograms, and `from_base_path()` to resolve path issues.

## Notes

Ensure to create a `.pth` file in the virtual environment's `/lib/site-packages/` to avoid import errors. Example contents for `src.pth`:
  - Unix: `absolute/path/to/cs334-project/src`
  - Windows: `Drive:\Users\User\path\to\cs334-project\src`