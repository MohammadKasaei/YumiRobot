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

from environment.yumiEnvSpatula import yumiEnvSpatula
        
import numpy as np



if __name__ == '__main__':

    env = yumiEnvSpatula()
    env.create_karolinska_env()
    time.sleep(5)

    # # Example usage
    # start_pos = 0.0
    # end_pos = 10.0
    # start_vel = 0.0
    # end_vel = 0.0
    # start_acc = 0.0
    # end_acc = 0.0
    # duration = 5.0
    # dt = 0.1

    # t, pos, vel, acc = env.fifth_order_trajectory_planner(start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)


    # Example usage
    # start_pos = np.array([0.0, 0.0, 0.0])
    # end_pos = np.array([10.0, 5.0, 3.0])
    # start_vel = np.array([1.0, 0.0, 0.0])
    # end_vel = np.array([0.0, 0.0, 1.0])
    # start_acc = np.array([0.0, 0.0, 0.0])
    # end_acc = np.array([0.0, 0.0, 0.0])
    # duration = 5.0
    # dt = 0.1

    # t, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc = env.fifth_order_trajectory_planner_3d(start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)



   
    gw = 0
    gt = 0
    x0 = np.array([0., 0.0 , 0.25])

    # time.sleep(10)
    env.reset_robot()
    env.wait(20)
    state = -1     
    target_rack = 1    

    while (True):


        if state == -1:            
            env.go_home()
            state = 0

        elif state == 0: # move on top of the box
            env.go_on_top_of_box()
            env.wait(1)
            # time.sleep(2)
            state = 2
        
        elif state == 2: # move inside the box
            env.go_inside_box()
            env.wait(1)
            state = 4

        elif state == 4: # grasp
            env.grasp()
            env.wait(5)
            state = 5
        
        elif state == 5: # lift up
            env.lift_up()
            state = 6
        
        elif state == 6: # move the rack towards the robot
            env.move_racks_to_station()
            state = 7

        elif state == 7: # put down
            env.place_racks_to_station()
            env.release_racks()
            env.release_arms()
            env.go_home()

            state = 10   #stop

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
    