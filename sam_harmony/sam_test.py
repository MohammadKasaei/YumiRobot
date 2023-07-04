import os
import cv2
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np

from PIL import Image
from segment_anything import sam_model_registry 
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from torch.utils.data import Dataset, DataLoader
import glob
import torch
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

# sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
# predictor = SamPredictor(sam)

sam_checkpoint = 'models/sam_vit_b_01ec64.pth'
model_type = "vit_b" #
# sam_checkpoint = 'models/sam_vit_l_0b3195.pth'
# model_type = "vit_b" 

device = "cuda" #cpu,cuda

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator1 = SamAutomaticMaskGenerator(sam, points_per_batch=16)
predictor = SamPredictor(sam)

image_path = "/home/mohammad/viT-Test/images/racks/racks.jpg"
image_information = cv2.imread(image_path)
image = cv2.resize(image_information, (224,224))

# fig, ax = plt.subplots(1,3)
# ax[0].imshow(image)
# # ax[1].imshow(cv2.merge(inputs.pixel_values.squeeze().numpy()) )
# # ax[2].imshow(cv2.merge(latent_vect[50].numpy()) )
# # print (last_hidden_states.shape)

# plt.show()

masks = mask_generator1.generate(image)



print ("salam")