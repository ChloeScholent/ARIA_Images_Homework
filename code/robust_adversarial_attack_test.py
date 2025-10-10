import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model_class import MNISTCNN
from adversarial_pattern import fgsm_attack, denorm, test_per_class
from dataset import adversarial_test_loader

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', device)

model_path = "Models/Robust_MNIST_CNN_Classification.pth"

print("Loading model...")

model = MNISTCNN(num_classes=10)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("Model sucessfully loaded !\n")

epsilons = [0, .005, .01, .05, .075, .1, .3]

# Testing FGSM attack
test_loader = adversarial_test_loader
accuracies_per_class = []
adv_examples_all = []

for eps in epsilons:
    accs, adv_examples = test_per_class(model, device, test_loader, eps)
    accuracies_per_class.append(accs)
    adv_examples_all.append(adv_examples)

accuracies_per_class = np.array(accuracies_per_class).T

# Plot accuracy per class
plt.figure(figsize=(8, 5))
for i in range(10):
    plt.plot(epsilons, accuracies_per_class[i], marker='o', label=f'Classe {i}')

plt.title("Précision par classe selon le bruit (ε)")
plt.xlabel("ε")
plt.ylabel("Précision")
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Plot several examples of adversarial samples at each epsilon
examples = adv_examples_all
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




