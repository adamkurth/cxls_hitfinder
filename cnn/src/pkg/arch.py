import torch
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from torch.autograd import Variable
from models import *


#architecture
models = get_models()
model = models.get('BasicCNN1')
channels, height, width = 1, 2163, 2069


# IMPLEMENT TO GENERATE CODE THAT GIVES US DIAGRAMS USING 
# https://keras.io/api/models/ 