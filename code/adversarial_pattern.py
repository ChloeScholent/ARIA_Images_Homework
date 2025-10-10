import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]):

    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test_per_class(model, device, test_loader, epsilon):
    model.eval()
    n_classes = 10  # pour MNIST
    correct = 0
    correct_per_class = np.zeros(n_classes)
    total_per_class = np.zeros(n_classes)
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue  # on ne perturbe que les exemples correctement classés

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        #Génération du bruit
        data_denorm = denorm(data)
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        output = model(perturbed_data_normalized)
        final_pred = output.max(1, keepdim=True)[1]

        label = target.item()
        total_per_class[label] += 1
        if final_pred.item() == label:
            correct_per_class[label] += 1
            correct +=1

        if len(adv_examples) < 5:  # on garde 5 exemples max pour chaque epsilon
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    acc_per_class = np.zeros(n_classes)
    for i in range(n_classes):
        if total_per_class[i] > 0:
            acc_per_class[i] = correct_per_class[i] / total_per_class[i]
        else:
            acc_per_class[i] = np.nan  # aucune image correctement classée avant attaque

    mean_acc = np.nanmean(acc_per_class)
    print(f"Epsilon: {epsilon:.4f} | Test Accuracy = {correct} / {float(len(test_loader))} = {correct/float(len(test_loader))*100:.2f}%")

    return acc_per_class, adv_examples