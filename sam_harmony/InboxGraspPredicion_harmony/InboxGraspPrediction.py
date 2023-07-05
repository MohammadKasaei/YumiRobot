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


class InboxGraspPrediction():
    def __init__(self,sam_model= "vit_b", device = "cuda" ) -> None:

        self._sam_model_type = sam_model
        self._device = device
        if self._sam_model_type == "vit_b":   # 375 MB         
            self._sam_checkpoint = 'sam_harmony/models/sam_vit_b_01ec64.pth'
        elif self._sam_model_type == "vit_h": # 2.6 GB           
            self._sam_checkpoint = 'sam_harmony/models/sam_vit_h_4b8939.pth'
        else: #1.2 GB
            self._sam_checkpoint = 'sam_harmony/models/sam_vit_l_0b3195.pth'

        self._device = device

        self._sam = sam_model_registry[self._sam_model_type](checkpoint=self._sam_checkpoint)
        self._sam.to(device=self._device)

        self._mask_generator1 = SamAutomaticMaskGenerator(self._sam, points_per_batch=16)
        self._predictor = SamPredictor(self._sam)
        self.config()


    def show_mask(self,mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(self,coords, labels, ax, marker_size=25):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='green', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='red', linewidth=1.25)   
        
    def show_box(self,box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

    def config(self):
        # input_point1 = np.array([300,200]).reshape(1,2)
        # input_point2 = np.array([300,260]).reshape(1,2)
        # step_x = 30
        # step_y = 10
        # for i in range(3):
        #     for j in range(3):
        #         input_point1 = np.vstack((input_point1,(input_point1[0,0]+i*step_x,input_point1[0,1]+j*step_y)))
        #         input_point2 = np.vstack((input_point2,(input_point2[0,0]+i*step_x,input_point2[0,1]+j*step_y)))
        # self._input_point = np.vstack((input_point1,input_point2))
        # self._input_label = np.ones(20)

        input_point1 = np.array([330,210]).reshape(1,2)
        step_x = 10
        step_y = 12
        for i in range(3):
            for j in range(7):
                input_point1 = np.vstack((input_point1,(input_point1[0,0]+i*step_x,input_point1[0,1]+j*step_y)))
        
        self._input_point = input_point1        
        self._input_label = np.ones(22)
        
        # remove regions
        # remove_point_start = np.array([200,100]).reshape(1,2)
        # step_x = 50
        # step_y = 50
        # for i in range(6):
        #     for j in range(8):
        #         input_point1 = np.vstack((input_point1,(remove_point_start[0,0]+i*step_x,remove_point_start[0,1]+j*step_y)))


        # self._input_point = input_point1        
        # self._input_label = np.ones(len(input_point1))

        # self._input_label[13:] *= 0



        # centre_point = np.array([330,240]).reshape(1,2)
        # lim_x = 20
        # lim_y = 20
        # for i in range(9):
        #         centre_point = np.vstack((centre_point,(centre_point[0,0]+int(np.random.uniform(-lim_x,lim_x)),centre_point[0,1]+int(np.random.uniform(-lim_y,lim_y)))))
        # input_point1 = centre_point
        # self._input_point = np.vstack((input_point1,input_point2))
        # self._input_label = np.ones(11)
        

    def generate_masks(self,image_path):
        self.image = cv2.imread(image_path)

        self.image_raw = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        kernel = np.ones((5,5),np.float32) / 30
        self.image = cv2.filter2D(self.image_raw,-1,kernel)
        
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        dilatation_size = 2
        # dilation_shape = cv2.MORPH_RECT
        dilation_shape = cv2.MORPH_ELLIPSE        
        element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                           (dilatation_size, dilatation_size))        
        self.image = cv2.dilate(self.image, element)


        self._predictor.set_image(self.image)
        self._masks, self._scores, self._logits = self._predictor.predict(
            point_coords=self._input_point,
            point_labels=self._input_label,
            # multimask_output=True,
            # box=input_box[None, :],
            multimask_output=True,
        )

        mask_input = self._logits[np.argmax(self._scores), :, :]  # Choose the model's best mask

        self._masks, _, _ = self._predictor.predict(
            point_coords=self._input_point,
            point_labels=self._input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

        return self._masks, self._scores
    
    def generate_grasp(self,mask,vis=True):

        (contours, hierarchy) = cv2.findContours(np.uint8(mask*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = 3 
        j = 0
        grasp_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                center = np.int16((np.mean(box[:,0]),np.mean(box[:,1])))
                tmp = box.tolist() 
                bs =np.array(sorted(tmp, key=lambda a_entry: a_entry[1]))
                center1 = np.int16(((bs[0,0]+bs[1,0])/2,(bs[0,1]+bs[1,1])/2))        
                center2 = np.int16(((bs[2,0]+bs[3,0])/2,(bs[2,1]+bs[3,1])/2))        
                grasp_list.append ([center,center1,center2])
                if vis:
                    cv2.drawContours(self.image, contours, j, (255, 255, 0), thickness)                
                    cv2.drawContours(self.image,[box],0,(0,255,255),thickness)
                    cv2.circle(self.image, center=center, radius=10, color = (255,255,255), thickness=-1) 
                    cv2.circle(self.image, center=center1, radius=10, color = (255,0,255), thickness=5) 
                    cv2.circle(self.image, center=center2, radius=10, color = (255,0,255), thickness=5) 
            j += 1

        return grasp_list
    

if __name__ == "__main__":

    gs = InboxGraspPrediction()    
    
    # image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230704-151403.png"
    # image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-090410.png"
    # image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-094624.png"
    # image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-095040.png"
    # image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-101956.png"
    # image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-101841.png"
    # image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-101714.png"
    # image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-101610.png"
    # image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-110838.png"
    image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-110838.png"
    
    masks, scores = gs.generate_masks(image_path)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.subplot(121)
        plt.imshow(gs.image_raw)
        plt.title('Original')
        plt.axis('on')
        plt.subplot(122)
        plt.imshow(gs.image)
        plt.title('Grasp')
        gs.show_mask(mask, plt.gca(),random_color=False)
        gs.show_points(gs._input_point, gs._input_label, plt.gca())
        gs_list = gs.generate_grasp(mask,vis=True)
        print ("grasp list:\n", gs_list)
        plt.imshow(gs.image)
        plt.axis('on')
        plt.show()

    print ("done")