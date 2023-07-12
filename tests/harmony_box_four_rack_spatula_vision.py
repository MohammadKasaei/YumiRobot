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
import matplotlib.pyplot as plt


if __name__ == '__main__':

    env = yumiEnvSpatula()
    obj_detection = InboxGraspPrediction() 
    env.create_karolinska_env()
    time.sleep(3)
    env.reset_robot()

    env.wait(20)
    state = -1     
    target_rack_level = 1    
    vis = False

    while (True):

        if state == -1:            
            env.go_home()
            state = 0

        elif state == 0: # move on top of the box
            if target_rack_level == 1:
                rgb, depth = env.capture_image()
                box_centre = env.find_box_centre(rgb,vis_output=vis) # detect the box 
                obj_detection.config_params_based_on_box_centre(box_centre)
                masks, scores = obj_detection.generate_masks2(rgb)
                for i, (mask, score) in enumerate(zip(masks, scores)):                    
                    gs_list = obj_detection.generate_grasp(mask,vis=True)
                    print ("grasp list:\n", gs_list)
                    box_center_pos = gs_list[0][0]            
                    env.pos_offset = env.convert_pixel_to_metter(box_center_pos) /1000.0
                    
                    env.box_ori    = np.pi/2 - np.arctan2(gs_list[0][2][1]-gs_list[0][1][1],gs_list[0][2][0]-gs_list[0][1][0])
                    print (f"offset: {env.pos_offset} ")
                    print (f"orientation: {env.box_ori:3.3f}")
                    
                    env.pos_offset = env.pos_offset @ np.array([[np.cos(env.box_ori),-np.sin(env.box_ori)],[np.sin(env.box_ori),np.cos(env.box_ori)]])


                    
                
                    # plt.subplot(121)
                    # plt.imshow(obj_detection.image_raw)
                    # plt.title('Original')
                    # plt.axis('on')
                    # plt.subplot(122)
                    if vis:
                        plt.imshow(obj_detection.image)
                        plt.title('Grasp')
                        obj_detection.show_mask(mask, plt.gca(),random_color=False)
                        obj_detection.show_points(obj_detection._input_point, obj_detection._input_label, plt.gca())
                        plt.imshow(obj_detection.image)
                        plt.axis('on')
                        plt.pause(1)
                
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

