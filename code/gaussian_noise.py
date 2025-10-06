import torch
import torchvision.transforms.v2 as T  # v2 transforms
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt


noise = T.GaussianNoise(mean=0.0, sigma=0.2, clip=True)

# apply (returns same dtype/shape)
noisy = noise()   # shape: (1,H,W)


