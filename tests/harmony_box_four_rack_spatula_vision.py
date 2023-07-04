import time
import numpy as np
import pybullet as p
import pybullet_data
import sys

import cv2
import random
from collections import namedtuple
from operator import methodcaller

# from environment.camera.camera import Camera, CameraIntrinsic
# from graspGenerator.grasp_generator import GraspGenerator

from environment.yumiEnvSpatula import yumiEnvSpatula
from sam_harmony.InboxGraspPredicion_harmony.InboxGraspPrediction import InboxGraspPrediction

        
import numpy as np


if __name__ == '__main__':

    env = yumiEnvSpatula()
    obj_detection = InboxGraspPrediction() 
    env.create_karolinska_env()
    time.sleep(5)
    env.reset_robot()
    env.wait(20)
    state = -1     
    target_rack_level = 1    

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

