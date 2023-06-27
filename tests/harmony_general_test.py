import time
import numpy as np
import pybullet as p
import pybullet_data
import sys
import cv2
import random

from collections import namedtuple
from operator import methodcaller

from environment.camera.camera import Camera, CameraIntrinsic
from graspGenerator.grasp_generator import GraspGenerator

from environment.yumiEnv import YumiEnv
        

if __name__ == '__main__':

    env = YumiEnv()
    env.creat_pile_of_cube(1)
    for i in range(10):
        env.add_a_cube(pos=[0.62+(np.random.randint(-3,3)/10.0),0.31,0.05],
                        size=[0.04,0.04,0.04],color=[i/10.0,0.5,i/10.0,1])
        
    networkName = "GGCNN"
    if (networkName == "GGCNN"):
            ##### GGCNN #####
            network_model = "GGCNN"           
            network_path = 'trained_models/GGCNN/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
            sys.path.append('trained_models/GGCNN')
    elif (networkName == "GR_ConvNet"):
            ##### GR-ConvNet #####
            network_model = "GR_ConvNet"           
            network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
  
    depth_radius = 2
    # env = BaiscEnvironment(GUI = True,robotType ="Panda",img_size= IMG_SIZE)
    # # env = BaiscEnvironment(GUI = True,robotType ="UR5",img_size= IMG_SIZE)
    # env.createTempBox(0.35, 2)
    # env.updateBackgroundImage(1)
    
    # gg = GraspGenerator(network_path, env.camera, depth_radius, env.camera.width, network_model)
    # env.creat_pile_of_tubes(10)
   

    gw = i = 0 

    while (True):
        gw = 1 if gw == 0 else 0

        # rgb ,depth = env.capture_image(0)    
        # number_of_predict = 1
        # output = True
        # grasps, save_name = gg.predict_grasp( rgb, depth, n_grasps=number_of_predict, show_output=output)
        # print(grasps)
        # if (grasps == []):
        #     print ("can not predict any grasp point")
        # else:
        #     env.visualize_predicted_grasp(grasps,color=[1,0,1],visibleTime=1)   

        env.move_left_gripper (gw=gw)
        env.move_right_gripper (gw=gw)
        
        env._dummy_sim_step(10)

        T  = 1
        w  = 2*np.pi/T
        radius = 0.1
        x0 = np.array([0.4,0,0.3])
        gt = 0

        for i in range(200):    
            gt += 0.01
            if i % 10 == 0:
                env.capture_image()
            ori = p.getQuaternionFromEuler([0,np.pi,0])
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.00*gt)))
            
            # pose = [[0.3,0.4-(i*0.006),0.3],ori]            
            pose = [xd,ori]
            env.move_left_arm(pose=pose)
            env._dummy_sim_step(50)
            # time.sleep(0.01)

        env.go_home()

        gt = 0
        for i in range(200):    
            gt += 0.01
            if i % 10 == 0:
                env.capture_image()
            ori = p.getQuaternionFromEuler([0,np.pi,0])
            # pose = [[0.3+(i*0.006),-0.4+(i*0.006),0.6],ori]            
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.00*gt)))
            pose = [xd,ori]
            env.move_right_arm(pose=pose)
            env._dummy_sim_step(50)
            # time.sleep(0.01)
        
        env.go_home()    
        time.sleep(0.1)
    