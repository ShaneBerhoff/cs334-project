import torch.nn as nn
import torch
import time

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs, device):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  model.train() # Set model to training mode
  # Repeat for each epoch
  for epoch in range(num_epochs):
    epoch_start = time.time()
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, (inputs, labels) in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += labels.size(0)

        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    epoch_duration = time.time() - epoch_start
    print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f} Duration: {epoch_duration:.2f} sec')

  print('Finished Training')