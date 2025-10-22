This repository aims at implementing a Fast Gradient Sign Attack on a classification CNN on the MNIST dataset.

Here is a description of the various python files:

- `model_class.py` defines the CNN model and the accuracy function
- `dataset.py` implements the various datasets used throughout the training and testing phases
- `adversarial_pattern.py` defines the FGSM function
- `classic_training.py` corresponds to the training of the model on the classic train_loader from MNIST
- `adversarial_training` corresponds to the training of a new robust CNN model with a combination of classic and perturbed MNIST train_loader
- `adversarial_attack_test.py` tests the classic CNN on a perturbed test_loader
- `robust_adversarial_attack_test.py` tests the robust CNN on a perturbed test_loader
- `test_gaussian_noise.py` tests the classic model with a noisy test_loader

