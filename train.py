import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from model import MyResNet50
from dataloader import make_train_dataloader

import os
import copy
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

epochs = 100
learning_rate = 0.01

base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "plant-seedlings-classification", "train")
weight_path = os.path.join(base_path, "weights", "weight.pth")
os.makedirs(os.path.dirname(weight_path), exist_ok=True)
result_path = os.path.join(base_path, "result")
os.makedirs(result_path, exist_ok=True)

train_loader, valid_loader = make_train_dataloader(train_data_path)

model = MyResNet50()
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()


train_loss_list = [0]
valid_loss_list = [0]
train_accuracy_list = [0]
valid_accuracy_list = [0]

best = 100
best_model_wts = copy.deepcopy(model.state_dict())
best_confusion_matrix = None

# train
for epoch in range(epochs):
    print(f"\nEpoch: {epoch+1}/{epochs}")
    print("-" * len(f"Epoch: {epoch+1}/{epochs}"))
    train_loss, valid_loss = 0.0, 0.0
    train_correct, valid_correct = 0, 0
    train_accuracy, valid_accuracy = 0.0, 0.0

    model.train()
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, preds = torch.max(output.data, 1)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        train_correct += torch.sum(preds == target.data)

    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_accuracy = float(train_correct) / len(train_loader.dataset)
    train_accuracy_list.append((train_accuracy))

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(valid_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            _, preds = torch.max(output.data, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            valid_loss += loss.item() * data.size(0)
            valid_correct += torch.sum(preds == target.data)

        valid_loss /= len(valid_loader.dataset)
        valid_loss_list.append(valid_loss)
        valid_accuracy = float(valid_correct) / len(valid_loader.dataset)
        valid_accuracy_list.append((valid_accuracy))

    print(f"Training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}")
    print(
        f"Training accuracy: {train_accuracy:.4f}, validation accuracy: {valid_accuracy:.4f}"
    )

    if valid_loss < best:
        best = valid_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        best_confusion_matrix = confusion_matrix(all_labels, all_preds)

torch.save(best_model_wts, weight_path)


print("\nFinished Training")


pd.DataFrame({"train-loss": train_loss_list, "valid-loss": valid_loss_list}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1,epoch+1)
plt.xlabel("Epoch"), plt.ylabel("Loss")
plt.savefig(os.path.join(result_path, "Loss_curve"))


pd.DataFrame(
    {"train-accuracy": train_accuracy_list, "valid-accuracy": valid_accuracy_list}
).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1,epoch+1)
plt.xlabel("Epoch"), plt.ylabel("Accuracy")
plt.savefig(os.path.join(result_path, "Training_accuracy"))

plt.figure(figsize=(10, 8))
sns.heatmap(best_confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(result_path, "Confusion_Matrix"))
