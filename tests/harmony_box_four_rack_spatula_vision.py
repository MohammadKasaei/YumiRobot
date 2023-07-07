import time
import numpy as np
import pybullet as p
import pybullet_data
import sys

import cv2
import random
from collections import namedtuple
from operator import methodcaller


from environment.yumiEnvSpatula import yumiEnvSpatula
from sam_harmony.InboxGraspPredicion_harmony.InboxGraspPrediction import InboxGraspPrediction

        
import numpy as np


if __name__ == '__main__':

    env = yumiEnvSpatula()
    # obj_detection = InboxGraspPrediction() 
    # env.create_karolinska_env()
    # env.add_chessboard(np.array([1,1,0.1]))
    xd = env.covert_pixel_to_robot_frame([340,210])
    id = env.add_a_cube(pos=[xd[0],xd[1],0.04],size=[0.02,0.02,0.02],color=[0.1,0.1,0.1,1],mass=1)

    xd[2] = 0.4     

    ori = p.getQuaternionFromEuler([0,np.pi,0]) 
    env.move_left_arm(traget_pose=[xd,ori])

    


    
    # time.sleep(5)
    # env.reset_robot()
    while True:
        env.move_left_arm(traget_pose=[xd,ori])
        env._dummy_sim_step(1)
        rgb, depth = env.capture_image()
        image = np.ascontiguousarray(rgb, dtype=np.uint8)
        cv2.line(rgb, (0, 240), (640, 240), (0, 255, 0), thickness=2)
        cv2.line(rgb, (320, 0), (320, 480), (0, 255, 0), thickness=2)

        # p.resetBasePositionAndOrientation()

        cv2.imshow('image', rgb)

        cv2.waitKey(1)

    env.wait(20)
    state = -1     
    target_rack_level = 1    

    while (True):

        if state == -1:            
            env.go_home()
            state = 0

        elif state == 0: # move on top of the box
            rgb, depth = env.capture_image()
            box_pos = env.find_box_centre(rgb,vis_output=True)
            # print(box_pos)

            # detect the box 
            # update the sam params based on the box  
            # env.save_image(bgr)   
            
            
            env.go_on_top_of_box()
            env.wait(1)
            # time.sleep(2)
            state = 2
        
        elif state == 2: # move inside the box
            env.go_inside_box(target_rack_level)
            env.wait(1)
            state = 4

        elif state == 4: # grasp
            env.grasp(target_rack_level)
            env.wait(5)
            state = 5
        
        elif state == 5: # lift up
            env.lift_up()
            state = 6
        
        elif state == 6: # move the rack towards the robot
            env.move_racks_to_station(target_rack_level)
            state = 7

        elif state == 7: # put down
            env.place_racks_to_station(target_rack_level)
            env.release_racks(target_rack_level)
            env.release_arms(target_rack_level)
            env.go_home()

            if target_rack_level ==1:
                target_rack_level = 2
                state = 0
            else:    
                state = 10   #stop

            
        else:
            env._dummy_sim_step(50)

