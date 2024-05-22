import os
import torch.nn as nn
import torch
from torch.nn import init
import time
import util

CLASSES = 6

class Homebrew(nn.Module):
    def __init__(self, input_path=None, save_path=util.from_base_path("/Data/models/test-homebrew"), epoch_tuning=False):
        super().__init__()
        self.save_path = save_path
        self.epoch_tuning = epoch_tuning
        self.transform = None

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
        self.lin = nn.Linear(in_features=64, out_features=CLASSES)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        
        if input_path is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            self.load_state_dict(torch.load(input_path, map_location=device))

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
        return f"homebrew{'-etuning' if self.epoch_tuning else ''}"

def train(model, train_dl, val_dl, max_epochs, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if model.epoch_tuning:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                        steps_per_epoch=len(train_dl),
                                                        epochs=max_epochs, anneal_strategy='linear')
    else:
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

def predict(model, val_dl, final=False):
    correct_predictions = [0] * CLASSES
    total_predictions = [0] * CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_predictions[label] += 1
                total_predictions[label] += 1

    if final:
        accuracies = []
        for i in range(CLASSES):
            if total_predictions[i] > 0:
                accuracy = correct_predictions[i] / total_predictions[i]
                accuracies.append(accuracy)
            else:
                accuracies.append(0.0)

        for i in range(CLASSES):
            print(f"{util.class_map(i)} Accuracy: {accuracies[i]:.4f}, Total items: {total_predictions[i]}")

    overall_acc = sum(correct_predictions) / sum(total_predictions)
    print(f"\nOverall Accuracy: {overall_acc:.4f}, Total items: {sum(total_predictions)}")
