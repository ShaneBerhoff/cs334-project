import urllib.request
import timm
import torch.nn as nn
import torch
import time

CLASSES = 6 # 0 sad, 1 angry, 2 disgust, 3 fear, 4 happy, 5 neutral

class MobileNetV3TL(nn.Module):
    def __init__(self, path=None, full=True):
        super(MobileNetV3TL, self).__init__()
        self.full = full

        if path is not None:
            self.load_state_dict(torch.load(path))
        else:
            self.model = timm.create_model('mobilenetv3_large_100', pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False

            if full:
                self.model.classifier = nn.Sequential(
                    nn.Linear(self.model.classifier.in_features, 12), # TODO: Change 40 to a better number
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(12, CLASSES)
                )
            else:
                self.model.classifier = nn.Linear(self.model.classifier.in_features, CLASSES)

    def forward(self, x):
        return self.model(x)
    
    def save(self, path, epoch=0):
        torch.save(self.state_dict(), f"{path}/mnv3tl-{'full' if self.full else 'small'}-e{epoch}.pt")


def train(model, train_dl, max_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=len(train_dl),
                                                    epochs=max_epochs, anneal_strategy='linear')
    
    model.train()

    for epoch in range(max_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

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
        
        if epoch % 10 == 0:
                model.save("./", epoch+1)
        

def predict(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f"Accuracy: {acc:.4f}, Total items: {total_prediction}")


if __name__ == '__main__':
    import os
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    
    model = MobileNetV3TL()

    # load image
    dogfile = "dog.jpg"
    if not os.path.exists(dogfile):
        import urllib
        urllib.request.urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", dogfile)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    img = Image.open(dogfile).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    # predict
    with torch.no_grad():
        out = model(tensor)
    probas = torch.nn.functional.softmax(out[0], dim=0)
    print(probas.shape)

    # print top 5
    classfile = "imagenet_classes.txt"
    if not os.path.exists(classfile):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", classfile)
    with open(classfile) as f:
        classes = [line.strip() for line in f.readlines()]
    top5_prob, top5_catid = torch.topk(probas, 5)
    for i in range(5):
        print(f"Class: {classes[top5_catid[i]]}, Probability: {top5_prob[i].item()}")
