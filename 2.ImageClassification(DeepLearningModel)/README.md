# About this DeepLearning Model:
  We will build an front end application to upload the image and deeplearning model predicts the name of the object with acccuracy.
  
# Steps for building the Image classification model:
## 1. Image classification model using pretrained DL model
  1.1 Define deeplearning model  
  2.2 Preprocess the data   
  3.3 Get prediction  

## 1.1 Define deep learning model
```diff
# import required modules
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# import pytorch related modules
import torch
from torchvision import transforms
from torchvision.models import densenet121
```
```diff
# define pretrained DL model
model = densenet121(pretrained=True)

model.eval();
```
## 1.2 Preprocess data
