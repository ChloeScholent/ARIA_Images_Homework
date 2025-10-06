import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model_class import MNISTCNN
from dataset import gaussian_test_loader, test_loader

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', device)

model_path = "Models/MNIST_CNN_Classification.pth"

print("Loading model...")

model = MNISTCNN(num_classes=10)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("Model sucessfully loaded !\n")

all_preds = []
all_labels = []

with torch.inference_mode():
    for test_inputs, test_labels in test_loader:
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        test_outputs = MNIST_Model_CNN(test_inputs)
        preds = torch.argmax(test_outputs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(test_labels.cpu().numpy())

# Concatenate all predictions
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Print confusion matrix & classification report
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds))




