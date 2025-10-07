import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model_class import MNISTCNN
from dataset import test_loader
from sklearn.metrics import confusion_matrix, classification_report

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

        test_outputs = model(test_inputs)
        preds = torch.argmax(test_outputs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(test_labels.cpu().numpy())

# Concatenate all predictions
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Print confusion matrix & classification report
print("\nConfusion Matrix Clean:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report Clean:\n", classification_report(all_labels, all_preds))


gaussian_preds = []
gaussian_labels = []

with torch.inference_mode():
    for test_inputs, test_labels in test_loader:

        noise = torch.randn_like(test_inputs) * 0.6
        test_inputs = test_inputs + noise
        test_inputs = torch.clamp(test_inputs, 0., 1.)
        # use imgs_noisy for forward pass

        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        test_outputs = model(test_inputs)
        preds = torch.argmax(test_outputs, dim=1)

        gaussian_preds.append(preds.cpu().numpy())
        gaussian_labels.append(test_labels.cpu().numpy())

# Concatenate all predictions
gaussian_preds = np.concatenate(gaussian_preds)
gaussian_labels = np.concatenate(gaussian_labels)

# Print confusion matrix & classification report
print("\nConfusion Matrix Gaussian Noise:\n", confusion_matrix(gaussian_labels, gaussian_preds))
print("\nClassification Report Gaussian Noise:\n", classification_report(gaussian_labels, gaussian_preds))


