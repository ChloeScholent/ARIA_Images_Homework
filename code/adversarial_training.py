import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path
from model_class import MNISTCNN, accuracy_fn
from dataset import train_loader, test_loader
from adversarial_pattern import denorm, fgsm_attack

# TensorBoard
writer = SummaryWriter()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', device, '\n')

# Model
num_classes = 10
Robust_MNIST_Model_CNN = MNISTCNN(num_classes).to(device)
print(Robust_MNIST_Model_CNN, '\nModel created successfully!\n')

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Robust_MNIST_Model_CNN.parameters(), lr=0.001)

epochs = 6
epsilon = 0.05  # FGSM perturbation

print('Training...\n')

for epoch in range(epochs):
    losses = []
    accs = []

    for train_input, train_labels in train_loader:
        train_input = train_input.to(device)
        train_labels = train_labels.to(device)

        # Enable gradients on input for FGSM
        train_input.requires_grad = True

        # Forward pass on original data
        Robust_MNIST_Model_CNN.zero_grad()
        outputs = Robust_MNIST_Model_CNN(train_input)
        loss = loss_fn(outputs, train_labels)
        loss.backward()

        # Generate adversarial examples
        data_denorm = denorm(train_input)
        perturbed_data = fgsm_attack(data_denorm, epsilon=epsilon, data_grad=train_input.grad)
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Concatenate original + adversarial
        data_combined = torch.cat([train_input, perturbed_data_normalized], dim=0)
        labels_combined = torch.cat([train_labels, train_labels], dim=0)

        # Train on combined data
        Robust_MNIST_Model_CNN.train()
        optimizer.zero_grad()
        outputs_combined = Robust_MNIST_Model_CNN(data_combined)
        loss_combined = loss_fn(outputs_combined, labels_combined)
        loss_combined.backward()
        optimizer.step()

        # Record metrics
        losses.append(loss_combined.item())
        accs.append(accuracy_fn(outputs_combined, labels_combined))

    # Compute epoch metrics safely
    train_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
    train_acc = sum(accs) / len(accs) if len(accs) > 0 else 0.0

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)

    # Evaluation
    Robust_MNIST_Model_CNN.eval()
    test_losses = []
    test_accs = []
    with torch.inference_mode():
        for test_inputs, test_labels in test_loader:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)

            test_outputs = Robust_MNIST_Model_CNN(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            test_acc = accuracy_fn(test_outputs, test_labels)

            test_losses.append(test_loss.item())
            test_accs.append(test_acc)

    test_loss_epoch = sum(test_losses)/len(test_losses) if len(test_losses) > 0 else 0.0
    test_acc_epoch = sum(test_accs)/len(test_accs) if len(test_accs) > 0 else 0.0

    writer.add_scalar('Loss/Test', test_loss_epoch, epoch)
    writer.add_scalar('Accuracy/Test', test_acc_epoch, epoch)

    if epoch % 2 == 0:
        print(f'Epoch {epoch}: Train Loss={train_loss:.5f}, Train Acc={train_acc:.2f}%, '
              f'Test Loss={test_loss_epoch:.5f}, Test Acc={test_acc_epoch:.2f}%')

writer.flush()
writer.close()

# Confusion matrix & classification report
Robust_MNIST_Model_CNN.eval()
all_preds = []
all_labels = []

with torch.inference_mode():
    for test_inputs, test_labels in test_loader:
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        test_outputs = Robust_MNIST_Model_CNN(test_inputs)
        preds = torch.argmax(test_outputs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(test_labels.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds))

# # Save model
# MODEL_PATH = Path("Models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
# MODEL_SAVE_PATH = MODEL_PATH / "Robust_MNIST_CNN_Classification.pth"

# print(f'Saving model to {MODEL_SAVE_PATH}')
# torch.save(Robust_MNIST_Model_CNN.state_dict(), MODEL_SAVE_PATH)
