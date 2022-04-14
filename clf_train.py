# basic libraries
import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import pickle

# libraries to calculate and plot performance
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# libraries to build models
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable