import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.npyio import save
import torch.utils.data
from PIL import Image
from datetime import datetime
import pybullet as p

from network.hardware.device import get_device
from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.visualisation.plot import plot_results, save_results
from network.utils.dataset_processing.grasp import detect_grasps
import os
import cv2
import collections
from pathlib import Path


class GraspGenerator:

    def __init__(self, net_path, camera, depth_radius, imgWidth=224, network='GR_ConvNet', device='cpu'):

        if (device=='cpu'):
            self.net = torch.load(net_path, map_location=device)
            self.device = get_device(force_cpu=True)
        else:
            #self.net = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(1))
            #self.device = get_device()
            print ("GPU is not supported yet! :( -- continuing experiment on CPU!" )
            self.net = torch.load(net_path, map_location='cpu')
            self.device = get_device(force_cpu=True)

        self.near = camera.near
        self.far = camera.far
        self.depth_r = depth_radius
        self.cameraPos = np.array([camera.x,camera.y,camera.z])
        
        self.fig = plt.figure(figsize=(10, 10))

        self.network = network

        self.pixelToMeter= 0.62 / imgWidth
        # self.pixelToMeter= 0.17 / imgWidth

        self.PIX_CONVERSION = 277 * imgWidth/224

        self.img_width = imgWidth
        self.IMG_ROTATION = -np.pi * 0.5 
        self.CAM_ROTATION = 0
        self.MAX_GRASP = 0.1
        self.DIST_BACKGROUND = 1.115
        
    def grasp_to_robot_frame(self, grasp, depth_img):
        """
        return: x, y, z, roll, opening length gripper, object height
        """
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]

        # Get area of depth values around center pixel
        x_min = np.clip(x_p-self.depth_r, 0, self.img_width)
        x_max = np.clip(x_p+self.depth_r, 0, self.img_width)
        y_min = np.clip(y_p-self.depth_r, 0, self.img_width)
        y_max = np.clip(y_p+self.depth_r, 0, self.img_width)
        depth_values = depth_img[x_min:x_max, y_min:y_max]

        # Get minimum depth value from selected area
        z_p = np.amin(depth_values)

        # Convert pixels to meters
        x_p *= self.pixelToMeter
        y_p *= self.pixelToMeter
        # z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)
        

        img_center = self.img_width / 2 - 0.5
        imgOriginRelative = np.array([-img_center*self.pixelToMeter,img_center*self.pixelToMeter,0])
        imgWorldOrigin  = p.multiplyTransforms(self.cameraPos, p.getQuaternionFromEuler([0,0,0]), imgOriginRelative, p.getQuaternionFromEuler([0,0,self.IMG_ROTATION]))
        robot_xyz = p.multiplyTransforms(imgWorldOrigin[0],imgWorldOrigin[1], np.array([x_p,y_p,-z_p]), p.getQuaternionFromEuler([0,0,grasp.angle]))

        yaw = p.getEulerFromQuaternion(robot_xyz[1])[2]

        # Covert pixel width to gripper width
        opening_length = (grasp.length / int(self.MAX_GRASP *
                          self.PIX_CONVERSION)) * self.MAX_GRASP

        obj_height = self.DIST_BACKGROUND - z_p

        return robot_xyz[0][0], robot_xyz[0][1], robot_xyz[0][2], yaw, opening_length, obj_height

    def predict(self, rgb, depth, n_grasps=1, show_output=False):

        max_val = np.max(depth)
        depth = depth * (255 / max_val)
        depth = np.clip((depth - depth.mean())/175, -1, 1)
        
        if (self.network == 'GR_ConvNet'):
            ##### GR-ConvNet #####
            depth = np.expand_dims(np.array(depth), axis=2)
            img_data = CameraData(width=self.img_width, height=self.img_width)
            x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
        elif (self.network == 'GGCNN'): 
            ##### GG-CNN ####
            # # max_val = np.max(depth)
            # # depth = depth * (255 / max_val)
            # # depth = np.clip((depth - depth.mean())/175, -1, 1)
            x = torch.from_numpy(depth.reshape(1, 1, self.img_width, self.img_width).astype(np.float32))
        else:
            print("The selected network has not been implemented yet -- please choose another network!")
            exit() 

        with torch.no_grad():
            xc = x.to(self.device)

            if (self.network == 'GR_ConvNet'):
                ##### GR-ConvNet #####
                pred = self.net.predict(xc)
                # print (pred)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = post_process_output(pred['pos'],
                                                                pred['cos'],
                                                                pred['sin'],
                                                                pred['width'],
                                                                pixels_max_grasp)
            
            elif (self.network == 'GGCNN'): 
                ##### GG-CNN ####
                pred = self.net(xc)
                # print (pred[0])
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = post_process_output(pred[0],
                                                                pred[1],
                                                                pred[2],
                                                                pred[3],
                                                                pixels_max_grasp)
    
        
        
        save_name = None
        if show_output:
            #fig = plt.figure(figsize=(10, 10))
            im_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            plot = plot_results(self.fig,
                                rgb_img=im_bgr,
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                depth_img=depth,
                                no_grasps=3,
                                grasp_width_img=width_img)

            if not os.path.exists('network_output'):
                os.mkdir('network_output')
            time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            save_name = 'network_output/{}'.format(time)
            plot.savefig(save_name + '.png')
            plot.clf()

        grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
        return grasps, save_name

    def predict_grasp(self, rgb, depth, n_grasps=1, show_output=False):
        predictions, save_name = self.predict(rgb, depth, n_grasps=n_grasps, show_output=show_output)
        grasps = []
        for grasp in predictions:
            x, y, z, yaw, opening_len, obj_height = self.grasp_to_robot_frame(grasp, depth)
            grasps.append((x, y, z, yaw, opening_len, obj_height))

        return grasps, save_name
