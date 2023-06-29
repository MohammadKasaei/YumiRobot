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

from environment.yumiEnvLongFinger import yumiEnvLongFinger
        

if __name__ == '__main__':

    env = yumiEnvLongFinger()
    
    env.add_a_cube(pos=[0.6,0,0.02],size=[0.33,0.12,0.02],color=[0.1,0.1,0.1,1],mass=5)
    env.add_a_cube_without_collision(pos=[0.6,0,0.02],size=[0.34,0.275,0.02],color=[0.1,0.1,0.1,1])
    env.create_harmony_box(box_centre=[0.6,0.])
    
    env.add_a_rack(centre=[0.53,0.0,0.05])
    # env.add_a_rack(centre=[0.68,0.0,0.05],color=[0,1,0,1])
    # env.add_a_rack(centre=[0.67,0.0,0.05],color=[0,1,0,1])
    
    # env.visualize_camera_position()
    

    gw = 0
    gt = 0
    x0 = np.array([0., 0.0 , 0.25])

    time.sleep(10)

    state = 0 
    while (True):


        if state == 0 : # go home
            # env.go_home()
            ori = p.getQuaternionFromEuler([0,np.pi,0])        
            xd  = np.array([0.5,0.4,0.6])
            pose_l = [xd,ori]        
            env.move_left_arm(traget_pose=pose_l)
            # env.move_left_arm_lf(pose_l)
            
            
            xd  = [0.5,-0.4,0.6]
            pose_r = [xd,ori]
            env.move_right_arm(traget_pose=pose_r)

            env.wait(10)
            state = 1

        elif state == 1: # move on top of the box
            move_y = 0.25
            for i in range(1000):
                ori = p.getQuaternionFromEuler([0,np.pi,0])        
                # xd  = np.array([0.5,0.14,0.6])
                xd  = np.array([0.5,0.4-(i*move_y/1000.0),0.6])
                
                pose_l = [xd,ori]        
                env.move_left_arm(traget_pose=pose_l)
                
                xd  = [0.5,-0.4+(i*move_y/1000.0),0.6]
                pose_r = [xd,ori]
                env.move_right_arm(traget_pose=pose_r)

                env._dummy_sim_step(15)

            env.wait(10)
            time.sleep(2)
            state = 2
        
        elif state == 2: # move inside the box
            depth = 0.342
            for i in range(100):
                ori = p.getQuaternionFromEuler([0,np.pi,0])        
                xd  = np.array([0.53,0.13,0.6-i*depth/100.0])
                pose_l = [xd,ori]        
                env.move_left_arm(traget_pose=pose_l)
                
                xd  = [0.53,-0.13,0.6-i*depth/100.0]
                pose_r = [xd,ori]
                env.move_right_arm(traget_pose=pose_r)
                env._dummy_sim_step(100)

            env.wait(10)
            time.sleep(2)
            state = 3

        elif state == 3: # move to pre grasp
            for i in range(100):
                ori = p.getQuaternionFromEuler([0,np.pi,0])        
                xd  = np.array([0.53,0.13,0.258])
                pose_l = [xd,ori]        
                env.move_left_arm(traget_pose=pose_l)
                
                xd  = [0.53,-0.13,0.258]
                pose_r = [xd,ori]
                env.move_right_arm(traget_pose=pose_r)
                env._dummy_sim_step(10)
            env.wait(2)
            state = 4

        elif state == 4: # grasp
            grasp_width = 0.001
            for i in range(1000):            
                ori = p.getQuaternionFromEuler([0,np.pi,0])        
                xd  = np.array([0.53,0.13-(i*grasp_width/1000.0),0.258])
                pose_l = [xd,ori]        
                env.move_left_arm(traget_pose=pose_l)
                
                xd  = [0.53,-0.13+(i*grasp_width/1000.0),0.258]
                pose_r = [xd,ori]
                env.move_right_arm(traget_pose=pose_r)
                # env.wait(0.1)
                env._dummy_sim_step(10)
            env.wait(5)
            state = 5
        
        elif state == 5: # lift up
            lift = 0.3 
            grasp_width = 0.014
            for i in range(1000):                
                ori = p.getQuaternionFromEuler([0,np.pi,0])        
                xd  = np.array([0.53,0.13-grasp_width,0.25+(i*lift/1000)])
                pose_l = [xd,ori]        
                env.move_left_arm(traget_pose=pose_l)
                
                xd  = [0.53,-0.13+grasp_width,0.25+(i*lift/1000)]
                pose_r = [xd,ori]
                env.move_right_arm(traget_pose=pose_r)
                # env.wait(0.1)
                env._dummy_sim_step(10)
            state = 6
        
        elif state == 6: # lift up
            lift = 0.3 
            grasp_width = 0.014
            move_x = 0.25

            for i in range(1000):                
                ori = p.getQuaternionFromEuler([0,np.pi,0])        
                xd  = np.array([0.53-(i*move_x/1000.0),0.13-grasp_width,0.25+lift])
                pose_l = [xd,ori]        
                env.move_left_arm(traget_pose=pose_l)
                
                xd  = [0.53-(i*move_x/1000.0),-0.13+grasp_width,0.25+lift]
                pose_r = [xd,ori]
                env.move_right_arm(traget_pose=pose_r)
                # env.wait(0.1)
                env._dummy_sim_step(10)

            state = 7
        elif state == 7: # lift up
            lift = 0.3 
            grasp_width = 0.014
            move_x = 0.24
            put_down = 0.20

            for i in range(1000):                
                ori = p.getQuaternionFromEuler([0,np.pi,0])        
                xd  = np.array([0.53-move_x,0.13-grasp_width,0.25+lift-i*(put_down/1000)])
                pose_l = [xd,ori]        
                env.move_left_arm(traget_pose=pose_l)
                
                xd  = [0.53-move_x,-0.13+grasp_width,0.25+lift-i*(put_down/1000)]
                pose_r = [xd,ori]
                env.move_right_arm(traget_pose=pose_r)
                # env.wait(0.1)
                env._dummy_sim_step(10)

            state = 8   
        else:
            
            # ori = p.getQuaternionFromEuler([0,np.pi,0])        
            # xd  = np.array([0.5,0.15,0.25])
            # pose_l = [xd,ori]        
            # env.move_left_arm(pose=pose_l)
            
            # xd  = [0.5,-0.15,0.25]
            # pose_r = [xd,ori]
            # env.move_right_arm(pose=pose_r)
            
            env._dummy_sim_step(50)


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
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.25+0.00*gt)))
            
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
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.25+0.00*gt)))
            pose = [xd,ori]
            env.move_right_arm(pose=pose)
            env._dummy_sim_step(50)
            # time.sleep(0.01)
        
        env.go_home()    
        time.sleep(0.1)
    