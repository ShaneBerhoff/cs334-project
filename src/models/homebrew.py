import argparse
import os
import torch.nn as nn
import torch
from torch.nn import init
import time
from data_loader import get_loaders
from models.load_model import load_best_model
import util

# ----------------------------
# Audio Classification Model
# ----------------------------
class TuningAudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, input_path=None, save_path="./Data/models/homebrew"):
        super().__init__()
        self.save_path = save_path
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        
        if input_path is not None:
            self.load_state_dict(torch.load(input_path))
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
    
    def save(self, epoch=0):
        # Create directory if it doesn't exist
        weights_path = os.path.join(self.save_path, "Weights")
        os.makedirs(weights_path, exist_ok=True)
        # Construct path and save
        full_path = os.path.join(weights_path, f"{self.name()}-e{epoch}.pt")
        torch.save(self.state_dict(), full_path)

    def name(self):
        return f"homebrew"

def train(model, train_dl, val_dl, max_epochs, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=len(train_dl),
                                                    epochs=max_epochs, anneal_strategy='cos')
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        model.train()

        for _, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += labels.size(0)

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        epoch_duration = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, Duration: {epoch_duration:.4f}")
        
        # Validation
        val_loss = 0.0
        val_correct_prediction = 0
        val_total_prediction = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, prediction = torch.max(outputs, 1)
                val_correct_prediction += (prediction == labels).sum().item()
                val_total_prediction += labels.size(0)

        val_loss /= len(val_dl)
        val_acc = val_correct_prediction / val_total_prediction
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save(epoch+1)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

def predict(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f"Accuracy: {acc:.4f}, Total items: {total_prediction}")
    
def main(batch, workers, save_path, epoch):
    os.makedirs(save_path, exist_ok=True)
    model = TuningAudioClassifier(save_path=save_path)
    train_dl, test_dl, train_index, test_index = get_loaders(batch_size=batch, num_workers=workers, split_ratio=0.9, n_mels=128, n_fft=2048, hop_len=512)

    with open(f'{save_path}/train_indices.txt', 'w') as f:
        for index in train_index:
            f.write(f"{index}\n")

    with open(f'{save_path}/test_indices.txt', 'w') as f:
        for index in test_index:
            f.write(f"{index}\n")
    
    print(f"Running train and predict on {model.name()} with batch size {batch} and {workers} workers")

    # train
    start = time.time()
    train(model, train_dl, test_dl, max_epochs=epoch, patience=5)
    print(f"Train time: {time.time() - start}")

    # predict
    # CURRENTLY DOESN'T LOAD BEST EPOCH
    # model, _, test_dl = load_best_model(model_info, workers)
    # start = time.time()
    predict(model, test_dl)
    print(f"Predict time: {time.time() - start}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=32, help='Batch size')
    parser.add_argument('--workers', default=6, help='Workers to use for data loading')
    parser.add_argument('--max_epochs', default=20, help='Max epochs for training')
    parser.add_argument('--save', default="homebrew", help='Name of directory to save model weights')
    args = parser.parse_args()

    main(batch=int(args.batch), workers=int(args.workers), save_path=os.path.join(util.from_base_path("/Data/models"), args.save), epoch=int(args.max_epochs))