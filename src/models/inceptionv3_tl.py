import urllib.request
import timm
from timm.data import resolve_data_config
import torch.nn as nn
import torch
import time
import os
from timm.data.transforms_factory import create_transform
from torchvision.transforms import Compose, Resize, Lambda, Normalize

CLASSES = 6 # 0 sad, 1 angry, 2 disgust, 3 fear, 4 happy, 5 neutral
def repeat_channels(x):
    return x.expand(3, -1, -1)

# optimal load seems to be batch size 32, workers 6 - minimal fluctuation in CUDA usage
class InceptionV3TL(nn.Module):
    def __init__(self, input_path=None, save_path="./Data/models/model1"):
        super(InceptionV3TL, self).__init__()
        self.save_path = save_path

        if input_path is not None:
            self.model = timm.create_model('inception_v3', pretrained=False, num_classes=CLASSES)

            for name, param in self.model.named_parameters():
                if name.split(".")[0] not in ["Mixed_7a", "Mixed_7b", "Mixed_7c"]:
                    param.requires_grad = False

            self.model.load_state_dict(torch.load(input_path))
        else:
            self.model = timm.create_model('inception_v3', pretrained=True, num_classes=CLASSES)

            for name, param in self.model.named_parameters():
                if name.split(".")[0] not in ["Mixed_7a", "Mixed_7b", "Mixed_7c"]:
                    param.requires_grad = False
        
        self.config = resolve_data_config({}, model=self.model)
        self.transform = Compose([
            Resize(self.config['input_size'][1:]),
            Lambda(repeat_channels),  # Replicate the channel to simulate RGB
            Normalize(mean=self.config['mean'], std=self.config['std'])
        ])

    def forward(self, x):
        return self.model(x)
    
    def save(self, epoch=0):
        # Create directory if it doesn't exist
        weights_path = os.path.join(self.save_path, "Weights")
        os.makedirs(weights_path, exist_ok=True)
        # Construct path and save
        full_path = os.path.join(weights_path, f"inv3tl-e{epoch}.pt")
        torch.save(self.model.state_dict(), full_path)

    def name(self):
        return f"inv3tl"


def train(model, train_dl, val_dl, max_epochs, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=len(train_dl),
                                                    epochs=max_epochs, anneal_strategy='linear')
    
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
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_dl)
        print(f"Validation Loss: {val_loss:.4f}")

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
