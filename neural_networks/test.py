import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as tfs
from torch.utils.data import TensorDataset, DataLoader

print(torch.randn(128))