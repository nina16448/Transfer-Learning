import torch
import pandas as pd
from model import MyResNet50
from dataloader import make_test_dataloader

import os
from tqdm import tqdm

class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

def predict_test_data(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend([class_names[p] for p in predicted.cpu().numpy()])

    return predictions

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_path, "plant-seedlings-classification", "test")
weight_path = os.path.join(base_path, "weights", "weight.pth")


model = MyResNet50()
try:
    model.load_state_dict(torch.load(weight_path))
except FileNotFoundError:
    print(f"Weight file not found at {weight_path}")
    exit(1)
except Exception as e:
    print(f"Error loading model weights: {str(e)}")
    exit(1)
model = model.to(device)

test_loader = make_test_dataloader(test_data_path)

predictions = predict_test_data(model, test_loader)

dfDict = {
    'file': os.listdir(test_data_path),
    'species': predictions
}

df = pd.DataFrame(dfDict)

csv_file_path = os.path.join(base_path, "predictions.csv")
df.to_csv(csv_file_path, index=False)

print(f"Predictions saved to {csv_file_path}")