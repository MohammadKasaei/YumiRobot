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
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


model_type = "vit_h" #
sam_checkpoint = 'models/sam_vit_h_4b8939.pth'
# model_type = "vit_b" #
# sam_checkpoint = 'models/sam_vit_b_01ec64.pth'
# model_type = "vit_l" 
# sam_checkpoint = 'models/sam_vit_l_0b3195.pth'

device = "cuda" #cpu,cuda

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator1 = SamAutomaticMaskGenerator(sam, points_per_batch=16)
predictor = SamPredictor(sam)

image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230704-151403.png"
# image_information = cv2.imread(image_path)
# image = cv2.resize(image_information, (224,224))
image = cv2.imread(image_path)
image_raw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

kernel = np.ones((5,5),np.float32) / 30
image = cv2.filter2D(image_raw,-1,kernel)

input_point1 = np.array([400,500]).reshape(1,2)
input_point2 = np.array([400,700]).reshape(1,2)
step_x = 100
step_y = 50
for i in range(3):
    for j in range(3):
        input_point1 = np.vstack((input_point1,(input_point1[0,0]+i*step_x,input_point1[0,1]+j*step_y)))
        input_point2 = np.vstack((input_point2,(input_point2[0,0]+i*step_x,input_point2[0,1]+j*step_y)))

# centre_point = np.array([600,800]).reshape(1,2)
# lim_x = 50
# lim_y = 50
# for i in range(9):
#         centre_point = np.vstack((centre_point,(centre_point[0,0]+int(np.random.uniform(-lim_x,lim_x)),centre_point[0,1]+int(np.random.uniform(-lim_y,lim_y)))))
# input_point = centre_point

input_point = np.vstack((input_point1,input_point2))
# input_label1 = np.ones(10).reshape(10,1)
# input_label2 = 2*np.ones(10).reshape(10,1)
input_label = np.ones(20)
# input_label[0:10] = 0
# input_label[10:] = 0



predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    # multimask_output=True,
    # box=input_box[None, :],
    multimask_output=True,
)

mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)


for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.subplot(121),plt.imshow(image_raw),plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    plt.axis('on')
    plt.subplot(122),plt.imshow(image),plt.title('Grasp')
    # show_points(input_point, input_label, plt.gca())
    show_mask(mask, plt.gca(),random_color=False)
    (contours, hierarchy) = cv2.findContours(np.uint8(mask*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = 3 
    j = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            # cv2.drawContours(image, [contours[j]], j, (0, 255, 0), thickness)
            cv2.drawContours(image, contours, j, (0, 255, 0), thickness)
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image,[box],0,(0,255,255),5)
            center = np.int16((np.mean(box[:,0]),np.mean(box[:,1])))
            cv2.circle(image, center=center, radius=30, color = (255,255,255), thickness=-1) 

            tmp = box.tolist() 
            bs =np.array(sorted(tmp, key=lambda a_entry: a_entry[1]))

            center1 = np.int16(((bs[0,0]+bs[1,0])/2,(bs[0,1]+bs[1,1])/2))        
            cv2.circle(image, center=center1, radius=30, color = (255,0,255), thickness=10) 
            center2 = np.int16(((bs[2,0]+bs[3,0])/2,(bs[2,1]+bs[3,1])/2))        
            cv2.circle(image, center=center2, radius=30, color = (255,0,255), thickness=10) 
        j += 1
    
    plt.imshow(image)
    # plt.xticks([]), plt.yticks([])
    plt.axis('on')
    plt.show()


# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(5,5))
#     plt.imshow(image)
#     show_mask(mask, plt.gca(),random_color=False)
#     show_points(input_point, input_label, plt.gca())
#     # show_box(input_box, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('on')
#     plt.show()  


# masks = mask_generator1.generate(image)

# plt.figure(figsize=(5,5))
# plt.imshow(image)
# plt.axis('on')
# plt.show()

print ("done")