import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model_class import MNISTCNN
from dataset import test_loader
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', device)

model_path = "Models/MNIST_CNN_Classification.pth"

print("Loading model...")

model = MNISTCNN(num_classes=10)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("Model loaded successfully !\n")

noise_levels = [0, .1, .2, .5, .7, 1]
accuracies = []

with torch.inference_mode():
    for sigma in noise_levels:
        all_preds = []
        all_labels = []

        for test_inputs, test_labels in test_loader:

            noisy_inputs = test_inputs + torch.randn_like(test_inputs) * sigma
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            noisy_inputs = noisy_inputs.to(device)
            test_labels = test_labels.to(device)

            outputs = model(noisy_inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(test_labels.cpu().numpy())

        # Concatenate results
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Accuracy
        acc = accuracy_score(all_labels, all_preds)
        accuracies.append(acc)
        print(f"Noise sigma = {sigma}: Accuracy = {acc*100:.2f}")


plt.figure(figsize=(8,5))
plt.plot(noise_levels, accuracies, marker='o')
plt.title("Model Accuracy vs Gaussian Noise")
plt.xlabel("Gaussian Noise Standard Deviation (σ)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
