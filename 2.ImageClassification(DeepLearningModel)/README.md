# About this DeepLearning Model:
  We will build an front end application to upload the image and get the deeplearning model predicts the name of the object with acccuracy.
  
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
```diff
# load image using PIL
input_image = Image.open(filename)

# preprocess image according to the pretrained model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)

# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0) 

# pass input batch to the model
with torch.no_grad():
    output = model(input_batch)
 ```
 ## 1.3 Get prediction
 ```diff 
 pred = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
np.argmax(pred)
```
```diff
# download classes on which the model was trained on 
!wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
```
```diff
# get the prediction accuracy
print(classes[str(np.argmax(pred))][1], round(max(pred)*100, 2))
```
# 2. Deploying Image Classification model
1.1 Install required libraries  
1.2 Setup DL model using streamlit  
1.3 Deploy DL model on AWS/Colab/HF spaces    

## 1.1 Install required libraries
```diff
!pip install -q streamlit
!pip install -q pyngrok
```
## 1.2 Setup DL model using streamlit
```diff
%%writefile app.py

## create streamlit app

# import required libraries and modules
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121

import streamlit as st

# define prediction function
def predict(image):
    # load DL model
    model = densenet121(pretrained=True)

    model.eval()

    # load classes
    with open('imagenet_class_index.json', 'r') as f:
        classes = json.load(f)

    # preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # get prediction
    with torch.no_grad():
        output = model(input_batch)

    pred = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()

    # return confidence and label
    confidence = round(max(pred)*100, 2)
    label = classes[str(np.argmax(pred))][1]

    return confidence, label

# define image file uploader
image = st.file_uploader("Upload image here")

# define button for getting prediction
if image is not None and st.button("Get prediction"):
    # load image using PIL
    input_image = Image.open(image)

    # show image
    st.image(input_image, use_column_width=True)

    # get prediction
    confidence, label = predict(input_image)

    # print results
    "Model is", confidence, "% confident that this image is of a", label
  ```
## 1.3 Deploy DL model
  ```diff
  # run streamlit app
  !streamlit run app.py &>/dev/null&
  ```
  ```diff
  # make streamlit app available publicly
  from pyngrok import ngrok

  public_url = ngrok.connect('8501');

  public_url
```

# Model can be deployed on AWS/Colab/Flask/Hugging Spaces

## Hugging spaces model

