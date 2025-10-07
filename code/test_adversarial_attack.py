import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model_class import MNISTCNN
from adversarial_pattern import fgsm_attack, denorm, test
from dataset import adversarial_test_loader

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

epsilons = [0, .005, .01, .05, .075, .1, .3]

test_loader = adversarial_test_loader

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()



# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()




