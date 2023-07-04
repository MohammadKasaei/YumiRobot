import time
import numpy as np
import pybullet as p
import pybullet_data
import sys
import cv2
import random

from collections import namedtuple
from operator import methodcaller

<<<<<<< Updated upstream
from environment.camera.camera import Camera #, CameraIntrinsic
# from graspGenerator.grasp_generator import GraspGenerator
=======
from environment.camera.camera import Camera, CameraIntrinsic
from graspGenerator.grasp_generator import GraspGenerator
from PIL import Image
>>>>>>> Stashed changes



class yumiEnvSpatula():
    def __init__(self) -> None:
        self.simulationStepTime = 0.005
        self.vis = True

        p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.simulationStepTime)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=90, cameraPitch=-40, cameraTargetPosition=[0,0,0])

        # sim_arg = {'gui':'True'}
        self.plane_id = p.loadURDF('plane.urdf')
        # self.load_robot(urdf = 'urdfs/yumi_grippers.urdf',print_joint_info=True)
        self.load_robot(urdf = 'urdfs/yumi_grippers_spatula.urdf',print_joint_info=True)

        self._left_FK_offset = np.array([0,0.01,0.245])
        self._right_FK_offset = np.array([0,-0.01,0.245])
        
        
        
        self.reset_robot()        
        self._dummy_sim_step(1000)

        print("\n\n\nRobot is armed and ready to use...\n\n\n")
        print ('-'*40)
        
        camera_pos = np.array([-0.08, 0.0, 0.8])
        camera_target = np.array([0.6, 0, 0.0])        
        self._init_camera(camera_pos,camera_target)

    

    def fifth_order_trajectory_planner_3d(self,start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt):
        """
        Generates a fifth-order trajectory plan for 3D position considering velocity and acceleration,
        given the initial and final conditions.

        Args:
            start_pos (numpy.ndarray): Starting position as a 1D array of shape (3,) for (x, y, z).
            end_pos (numpy.ndarray): Ending position as a 1D array of shape (3,) for (x, y, z).
            start_vel (numpy.ndarray): Starting velocity as a 1D array of shape (3,) for (x, y, z).
            end_vel (numpy.ndarray): Ending velocity as a 1D array of shape (3,) for (x, y, z).
            start_acc (numpy.ndarray): Starting acceleration as a 1D array of shape (3,) for (x, y, z).
            end_acc (numpy.ndarray): Ending acceleration as a 1D array of shape (3,) for (x, y, z).
            duration (float): Desired duration of the trajectory.
            dt (float): Time step for the trajectory plan.

        Returns:
            tuple: A tuple containing time, position, velocity, and acceleration arrays for x, y, and z coordinates.
        """
        # Calculate the polynomial coefficients for each dimension
        t0 = 0.0
        t1 = duration
        t = np.arange(t0, t1, dt)
        n = len(t)

        A = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                    [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                    [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                    [1, t1, t1 ** 2, t1 ** 3, t1 ** 4, t1 ** 5],
                    [0, 1, 2 * t1, 3 * t1 ** 2, 4 * t1 ** 3, 5 * t1 ** 4],
                    [0, 0, 2, 6 * t1, 12 * t1 ** 2, 20 * t1 ** 3]])

        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        acc = np.zeros((n, 3))

        for dim in range(3):
            b_pos = np.array([start_pos[dim], start_vel[dim], start_acc[dim], end_pos[dim], end_vel[dim], end_acc[dim]])
            x_pos = np.linalg.solve(A, b_pos)

            # Generate trajectory for the dimension using the polynomial coefficients
            pos[:, dim] = np.polyval(x_pos[::-1], t)
            vel[:, dim] = np.polyval(np.polyder(x_pos[::-1]), t)
            acc[:, dim] = np.polyval(np.polyder(np.polyder(x_pos[::-1])), t)

        return t, pos[:, 0], pos[:, 1], pos[:, 2], vel[:, 0], vel[:, 1], vel[:, 2], acc[:, 0], acc[:, 1],acc[:, 2]


    def fifth_order_trajectory_planner(self,start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt):
        """
        Generates a fifth-order trajectory plan given the initial and final conditions.

        Args:
            start_pos (float): Starting position.
            end_pos (float): Ending position.
            start_vel (float): Starting velocity.
            end_vel (float): Ending velocity.
            start_acc (float): Starting acceleration.
            end_acc (float): Ending acceleration.
            duration (float): Desired duration of the trajectory.
            dt (float): Time step for the trajectory plan.

        Returns:
            tuple: A tuple containing time, position, velocity, and acceleration arrays.
        """
        # Calculate the polynomial coefficients
        t0 = 0.0
        t1 = duration
        t = np.arange(t0, t1, dt)
        n = len(t)

        A = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                    [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                    [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                    [1, t1, t1 ** 2, t1 ** 3, t1 ** 4, t1 ** 5],
                    [0, 1, 2 * t1, 3 * t1 ** 2, 4 * t1 ** 3, 5 * t1 ** 4],
                    [0, 0, 2, 6 * t1, 12 * t1 ** 2, 20 * t1 ** 3]])

        b_pos = np.array([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc])

        x_pos = np.linalg.solve(A, b_pos)

        # Generate trajectory using the polynomial coefficients
        pos = np.polyval(x_pos[::-1], t)
        vel = np.polyval(np.polyder(x_pos[::-1]), t)
        acc = np.polyval(np.polyder(np.polyder(x_pos[::-1])), t)

        return t, pos, vel, acc
    


    @staticmethod
    def _ang_in_mpi_ppi(angle):
        """
        Convert the angle to the range [-pi, pi).

        Args:
            angle (float): angle in radians.

        Returns:
            float: equivalent angle in [-pi, pi).
        """

        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle

    def load_robot (self,urdf, print_joint_info = False):        
        self.robot_id = p.loadURDF(urdf,[0, 0, -0.11], [0, 0, 0, 1])
        numJoints = p.getNumJoints(self.robot_id)
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        controlJoints = ["yumi_joint_1_r", "yumi_joint_2_r", "yumi_joint_7_r", "yumi_joint_3_r",
                         "yumi_joint_4_r", "yumi_joint_5_r", "yumi_joint_6_r","gripper_r_joint","gripper_r_joint_m",
                         "yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", "yumi_joint_3_l",
                         "yumi_joint_4_l", "yumi_joint_5_l", "yumi_joint_6_l","gripper_l_joint","gripper_l_joint_m"]
        
        self._left_ee_frame_name  = 'yumi_link_7_l_joint_3'
        self._right_ee_frame_name = 'yumi_link_7_r_joint_3'
        
        self._LEFT_HOME_POSITION = [-0.473, -1.450, 1.091, 0.031, 0.513, 0.77, -1.669]
        self._RIGHT_HOME_POSITION = [0.413, -1.325, -1.040, -0.053, -0.484, 0.841, 1.669]

        self._RIGHT_HAND_JOINT_IDS = [1,2,3,4,5,6,7]
        self._RIGHT_GRIP_JOINT_IDS = [8,9,10,11]
                
        self._LEFT_HAND_JOINT_IDS = [12,13,14,15,16,17,18]
        self._LEFT_GRIP_JOINT_IDS = [19,20,21,22]

        self._max_torques = [42, 90, 39, 42, 3, 12, 1]


        self._jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable", "jointAxis", "parentFramePos", "parentFrameOrn"])
        
        self._joint_Damping =  [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 
                                0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
        for i in range(numJoints):
            info = p.getJointInfo(self.robot_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            jointAxis = info[13]
            parentFramePos = info[14]
            parentFrameOrn = info[15]
            controllable = True if jointName in controlJoints else False
            info = self._jointInfo(jointID, jointName, jointType, jointLowerLimit,
                            jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable,
                            jointAxis, parentFramePos, parentFrameOrn)

            if info.type == "REVOLUTE" or info.type == "PRISMATIC" or True:  # set revolute joint to static
                p.setJointMotorControl2(self.robot_id, info.id, p.POSITION_CONTROL, targetPosition=0, force=300)
                if print_joint_info:
                    print (info)
                    print (jointType)                            
                    print ('-'*40)

    def _init_camera(self,camera_pos,camera_target,visulize_camera = False):        
        self._camera_pos = camera_pos
        self._camera_target = camera_target
        
        self.camera = Camera(cam_pos=self._camera_pos, cam_target= self._camera_target, near = 0.2, far = 2, size= [640, 480], fov=60)
        if visulize_camera:
            self.visualize_camera_position(self._camera_pos)

    def _dummy_sim_step(self,n):
        for _ in range(n):
            p.stepSimulation()
    
    def wait(self,sec):
        for _ in range(1+int(sec/self.simulationStepTime)):
            p.stepSimulation()

    def reset_robot(self):
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._LEFT_HAND_JOINT_IDS,targetPositions  = self._LEFT_HOME_POSITION)
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._RIGHT_HAND_JOINT_IDS,targetPositions = self._RIGHT_HOME_POSITION)
        self._dummy_sim_step(100)
    
    def go_home(self):
        p0,o0 = self.get_left_ee_state()
        start_pos = p0
        end_pos = np.array([0.1,0.4,0.3])
        start_vel = np.array([.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, .0])
        start_acc = np.array([0.0, 0.0, 0.0])
        end_acc = np.array([0.0, 0.0, 0.0])
        duration = 5.0
        dt = 0.005
        t, xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
        
        p0,o0 = self.get_right_ee_state()
        start_pos = p0
        end_pos = np.array([0.1,-0.4,0.3])
        t, xr_pos, yr_pos, zr_pos, xr_vel, yr_vel, zr_vel, xr_acc, yr_acc, zr_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
                    
        ori = p.getQuaternionFromEuler([0,np.pi,0])        

        for i in range(len(t)):
            xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
            pose_l = [xd,ori]      
            xd = np.array([xr_pos[i],yr_pos[i],zr_pos[i]])
            pose_r = [xd,ori]      
                
            self.move_left_arm(traget_pose=pose_l)
            self.move_right_arm(traget_pose=pose_r)
            
            self._dummy_sim_step(1)
        
    def go_on_top_of_box(self):
        p0,o0 = self.get_left_ee_state()
        start_pos = p0
        end_pos = np.array([0.5,0.145,0.5])
        start_vel = np.array([.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, .0])
        start_acc = np.array([0.0, 0.0, 0.0])
        end_acc = np.array([0.0, 0.0, 0.0])
        duration = 6.0
        dt = 0.005
        t, xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
        
        p0,o0 = self.get_right_ee_state()
        start_pos = p0
        end_pos = np.array([0.5,-0.145,0.5])
        t, xr_pos, yr_pos, zr_pos, xr_vel, yr_vel, zr_vel, xr_acc, yr_acc, zr_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
                    
        ori = p.getQuaternionFromEuler([0,np.pi,0])        

        for i in range(len(t)):
            xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
            pose_l = [xd,ori]      
            xd = np.array([xr_pos[i],yr_pos[i],zr_pos[i]])
            pose_r = [xd,ori]      
                
            self.move_left_arm(traget_pose=pose_l)
            self.move_right_arm(traget_pose=pose_r)
            
            self._dummy_sim_step(1)
                        
    def go_inside_box(self,racks_level):
        
        depth = 0.26 if racks_level == 2 else 0.33
        p0,o0 = self.get_left_ee_state()
        start_pos = p0
        end_pos = np.array([0.5,0.145,depth])
        start_vel = np.array([.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, .0])
        start_acc = np.array([0.0, 0.0, 0.0])
        end_acc = np.array([0.0, 0.0, 0.0])
        duration = 5.0
        dt = 0.005
        t, xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
        
        p0,o0 = self.get_right_ee_state()
        start_pos = p0
        end_pos = np.array([0.5,-0.145,depth])
        t, xr_pos, yr_pos, zr_pos, xr_vel, yr_vel, zr_vel, xr_acc, yr_acc, zr_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
                    
        ori = p.getQuaternionFromEuler([0,np.pi,0])        

        for i in range(len(t)):
            xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
            pose_l = [xd,ori]      
            xd = np.array([xr_pos[i],yr_pos[i],zr_pos[i]])
            pose_r = [xd,ori]      
                
            self.move_left_arm(traget_pose=pose_l)
            self.move_right_arm(traget_pose=pose_r)
            
            self._dummy_sim_step(1)

    def grasp(self,racks_level):
        depth =  0.26 if racks_level == 2 else 0.33
        grasp_width = 0.043
        p0,o0 = self.get_left_ee_state()
        start_pos = p0
        end_pos = np.array([0.5,0.145-grasp_width ,depth])
        start_vel = np.array([.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, .0])
        start_acc = np.array([0.0, 0.004, 0.0])
        end_acc = np.array([0.0, 0.004, 0.0])
        duration = 0.5
        dt = 0.005
        t, xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
        
        p0,o0 = self.get_right_ee_state()
        start_pos = p0
        end_pos = np.array([0.5,-0.145+grasp_width,depth])
        t, xr_pos, yr_pos, zr_pos, xr_vel, yr_vel, zr_vel, xr_acc, yr_acc, zr_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
                    
        ori = p.getQuaternionFromEuler([0,np.pi,0])        

        for i in range(len(t)):
            xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
            pose_l = [xd,ori]      
            xd = np.array([xr_pos[i],yr_pos[i],zr_pos[i]])
            pose_r = [xd,ori]      
                
            self.move_left_arm(traget_pose=pose_l)
            self.move_right_arm(traget_pose=pose_r)
            
            self._dummy_sim_step(1)
    
    def lift_up(self):
        grasp_width = 0.05
        lift_up = 0.5
        p0,o0 = self.get_left_ee_state()
        start_pos = p0
        end_pos = np.array([0.5,0.145-grasp_width ,lift_up])
        start_vel = np.array([.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, .0])
        start_acc = np.array([0.0, 0.0, 0.0])
        end_acc = np.array([0.0, 0.0, 0.0])
        duration = 5.0
        dt = 0.005
        t, xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
        
        p0,o0 = self.get_right_ee_state()
        start_pos = p0
        end_pos = np.array([0.5,-0.145+grasp_width,lift_up])
        t, xr_pos, yr_pos, zr_pos, xr_vel, yr_vel, zr_vel, xr_acc, yr_acc, zr_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
                    
        ori = p.getQuaternionFromEuler([0,np.pi,0])        

        for i in range(len(t)):
            xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
            pose_l = [xd,ori]      
            xd = np.array([xr_pos[i],yr_pos[i],zr_pos[i]])
            pose_r = [xd,ori]      
                
            self.move_left_arm(traget_pose=pose_l)
            self.move_right_arm(traget_pose=pose_r)
            
            self._dummy_sim_step(1)

    def move_racks_to_station(self,racks_level):
        station_x = 0.1 if racks_level== 1 else -0.05 
        p0,o0 = self.get_left_ee_state()
        start_pos = p0
        end_pos = np.array([-station_x,0.4,0.5])
        start_vel = np.array([.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, .0])
        start_acc = np.array([0.0, 0.0, 0.0])
        end_acc = np.array([0.0, 0.0, 0.0])
        duration = 45.0
        dt = 0.005

        t, xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
        
        p0,o0 = self.get_right_ee_state()
        start_pos = p0
        end_pos = np.array([-station_x,-0.45,0.5])        
        t, xr_pos, yr_pos, zr_pos, xr_vel, yr_vel, zr_vel, xr_acc, yr_acc, zr_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
                    
        ori = p.getQuaternionFromEuler([0,np.pi,0])        

        for i in range(len(t)):
            xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
            pose_l = [xd,ori]      
            xd = np.array([xr_pos[i],yr_pos[i],zr_pos[i]])
            pose_r = [xd,ori]      
                
            self.move_left_arm(traget_pose=pose_l)
            self.move_right_arm(traget_pose=pose_r)
            
            self._dummy_sim_step(1)         

    def place_racks_to_station(self,racks_level):
        
        station_x = 0.2 if racks_level== 1 else -0.05
        p0,o0 = self.get_left_ee_state()
        start_pos = p0
        end_pos = np.array([-station_x,0.33,0.33])
        start_vel = np.array([.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, .0])
        start_acc = np.array([0.0, 0.0, 0.0])
        end_acc = np.array([0.0, 0.0, 0.0])
        duration = 45.0
        dt = 0.005

        t, xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
        
        p0,o0 = self.get_right_ee_state()
        start_pos = p0
        end_pos = np.array([-station_x,-0.33,0.33])        
        t, xr_pos, yr_pos, zr_pos, xr_vel, yr_vel, zr_vel, xr_acc, yr_acc, zr_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
                    
        ori = p.getQuaternionFromEuler([0,np.pi,0])        

        for i in range(len(t)):
            xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
            pose_l = [xd,ori]      
            xd = np.array([xr_pos[i],yr_pos[i],zr_pos[i]])
            pose_r = [xd,ori]      
                
            self.move_left_arm(traget_pose=pose_l)
            self.move_right_arm(traget_pose=pose_r)
            
            self._dummy_sim_step(1)     

    def release_racks(self,racks_level):
        station_x = 0.2 if racks_level== 1 else -0.05
        p0,o0 = self.get_left_ee_state()
        start_pos = p0
        end_pos = np.array([-station_x,0.33,0.27])
        start_vel = np.array([.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, .0])
        start_acc = np.array([0.0, 0.0, 0.0])
        end_acc = np.array([0.0, 0.0, 0.0])
        duration = 5.0
        dt = 0.005

        t, xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
        
        p0,o0 = self.get_right_ee_state()
        start_pos = p0
        end_pos = np.array([-station_x,-0.33,0.27])        
        t, xr_pos, yr_pos, zr_pos, xr_vel, yr_vel, zr_vel, xr_acc, yr_acc, zr_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
                    
        ori = p.getQuaternionFromEuler([0,np.pi,0])        

        for i in range(len(t)):
            xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
            pose_l = [xd,ori]      
            xd = np.array([xr_pos[i],yr_pos[i],zr_pos[i]])
            pose_r = [xd,ori]      
                
            self.move_left_arm(traget_pose=pose_l)
            self.move_right_arm(traget_pose=pose_r)
            
            self._dummy_sim_step(1)                             
        

    def release_arms(self,racks_level):
        station_x = 0.2 if racks_level== 1 else -0.05
        p0,o0 = self.get_left_ee_state()
        start_pos = p0
        end_pos = np.array([-station_x,0.38,0.27])
        start_vel = np.array([.0, 0.0, 0.0])
        end_vel = np.array([0.0, 0.0, .0])
        start_acc = np.array([0.0, 0.0, 0.0])
        end_acc = np.array([0.0, 0.0, 0.0])
        duration = 5.0
        dt = 0.005

        t, xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
        
        p0,o0 = self.get_right_ee_state()
        start_pos = p0
        end_pos = np.array([-station_x,-0.38,0.27])        
        t, xr_pos, yr_pos, zr_pos, xr_vel, yr_vel, zr_vel, xr_acc, yr_acc, zr_acc = self.fifth_order_trajectory_planner_3d(
            start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt)
                    
        ori = p.getQuaternionFromEuler([0,np.pi,0])        

        for i in range(len(t)):
            xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
            pose_l = [xd,ori]      
            xd = np.array([xr_pos[i],yr_pos[i],zr_pos[i]])
            pose_r = [xd,ori]      
                
            self.move_left_arm(traget_pose=pose_l)
            self.move_right_arm(traget_pose=pose_r)
            
            self._dummy_sim_step(1)                             

                                        
    def get_left_ee_state(self):
        pose = p.getLinkState(self.robot_id,self._LEFT_GRIP_JOINT_IDS[-2],computeForwardKinematics=1)[0:2]        
        return pose[0]+self._left_FK_offset , pose[1]
    
    def get_right_ee_state(self):
        pose =  p.getLinkState(self.robot_id,self._RIGHT_GRIP_JOINT_IDS[-2],computeForwardKinematics=1)[0:2]
        return pose[0]+self._right_FK_offset , pose[1]


    def move_left_arm(self,traget_pose):                
       
        # ll = np.array([-2.8,-2.4,-0.1,-2.1,-5.0,-1.25,-3.9])
        # ul = np.array([ 2.8, 0.7, 0.1, 1.3, 5.0, 2.25,3.9])
        # jr = np.array([ 5.6,-1.4, 0.1,-0.8, 10.0,3.5,4.8])
        # jd = np.array([ 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        
        # joint_poses = p.calculateInverseKinematics(self.robot_id, 
        #                                            self._LEFT_HAND_JOINT_IDS[-1], 
        #                                            traget_pose[0], traget_pose[1],
        #                                            lowerLimits=ll,
        #                                            upperLimits=ul,
        #                                            jointRanges=jr)
        
        joint_poses = p.calculateInverseKinematics(self.robot_id, 
                                                   self._LEFT_HAND_JOINT_IDS[-1], 
                                                   traget_pose[0], traget_pose[1])

        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL,
                                    jointIndices = self._LEFT_HAND_JOINT_IDS,
                                    targetPositions  = joint_poses[7:14])
    
    def move_left_arm_lf(self,traget_pose):                
        p0 = self.get_left_ee_state()

        desired_pose = np.copy(traget_pose)
        desired_pose[0] = 0.95*np.array(p0[0])+0.05*traget_pose[0]
        desired_pose[1] = 0.*np.array(p0[1])+1*np.array(traget_pose[1])
        

        joint_poses = p.calculateInverseKinematics(self.robot_id, self._LEFT_HAND_JOINT_IDS[-1], desired_pose[0], desired_pose[1])
        joint_poses = list(map(self._ang_in_mpi_ppi, joint_poses))
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._LEFT_HAND_JOINT_IDS,targetPositions  = joint_poses[7:14])
        

    
    def move_right_arm_lf(self,traget_pose):                
        return 
        p0 = self.get_right_ee_state()

        desired_pose = np.copy(traget_pose)
        desired_pose[0] = 0.95*np.array(p0[0])+0.05*traget_pose[0]
        desired_pose[1] = 0.*np.array(p0[1])+1*np.array(traget_pose[1])
        

        joint_poses = p.calculateInverseKinematics(self.robot_id, self._RIGHT_HAND_JOINT_IDS[-1], desired_pose[0], desired_pose[1])
        joint_poses = list(map(self._ang_in_mpi_ppi, joint_poses))
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._RIGHT_HAND_JOINT_IDS,targetPositions  = joint_poses[:7], )

    def move_right_arm(self,traget_pose):        
        joint_poses = p.calculateInverseKinematics(self.robot_id, 
                                                   self._RIGHT_HAND_JOINT_IDS[-1], 
                                                   traget_pose[0], traget_pose[1])
        p.setJointMotorControlArray(self.robot_id,
                                    controlMode = p.POSITION_CONTROL, 
                                    jointIndices = self._RIGHT_HAND_JOINT_IDS,
                                    targetPositions  = joint_poses[:7] )
    
    
    def move_left_gripper(self, gw=0):
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._LEFT_GRIP_JOINT_IDS,targetPositions = [gw,gw])
        
    def move_right_gripper(self, gw=0):
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._RIGHT_GRIP_JOINT_IDS,targetPositions = [gw,gw])

  
    def add_a_cube(self,pos,size=[0.1,0.1,0.1],mass = 0.1, color = [1,1,0,1]):

        # cubesID = []
        box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = p.createMultiBody(mass, box, vis, pos, [0,0,0,1])
        p.changeDynamics(obj_id, 
                        -1,
                        spinningFriction=0.001,
                        rollingFriction=0.001,
                        linearDamping=0.0)
        # cubesID.append(obj_id)
        
        p.stepSimulation()
        return obj_id 
    
    def add_a_cube_without_collision(self,pos,size=[0.1,0.1,0.1], color = [1,1,0,1]):

        # cubesID = []
        box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0, 0, 0])
        vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = p.createMultiBody(0, box, vis, pos, [0,0,0,1])
        p.stepSimulation()
        return obj_id 
    
    def add_a_rack(self,centre, color = [1,0,0,1]):

        # cubesID = []
        size = [0.105, 0.205,0.05]
        mass = 0.1
        box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = p.createMultiBody(mass, box, vis, centre, [0,0,0,1])
        p.changeDynamics(obj_id, 
                        -1,
                        spinningFriction=0.001,
                        rollingFriction=0.001,
                        linearDamping=0.0)
        # cubesID.append(obj_id)
        p.stepSimulation()        
        return obj_id 
    

    def add_red_rack(self,centre):
        rack_width = 0.2
        # obj_id = p.loadURDF(f"objects/rack/urdf/rack_red.urdf",
        #                 [centre[0] - rack_width / 2.0, centre[1], centre[2]],
        #                 p.getQuaternionFromEuler([0, 0, np.pi/2]))
        obj_id = p.loadURDF(f"objects/rack/urdf/rack_red_with_tubes.urdf",
                        [centre[0] - rack_width / 2.0, centre[1], centre[2]],
                        p.getQuaternionFromEuler([0, 0, np.pi/2]))
        return obj_id
    

    def add_green_rack(self,centre):
        rack_width = 0.2
        # obj_id = p.loadURDF(f"objects/rack/urdf/rack_green.urdf",
        #                 [centre[0] - rack_width / 2.0, centre[1], centre[2]],
        #                 p.getQuaternionFromEuler([0, 0, np.pi/2]))
        obj_id = p.loadURDF(f"objects/rack/urdf/rack_green_with_tubes.urdf",
                        [centre[0] - rack_width / 2.0, centre[1], centre[2]],
                        p.getQuaternionFromEuler([0, 0, np.pi/2]))
        
        return obj_id

    def create_karolinska_env(self):
        
        self.add_a_cube_without_collision(pos=[0.5,0,0.002],size=[0.5,1,0.004],color=[0.9,0.9,0.9,1])

        self.create_harmony_box(box_centre=[0.5,0.])
        self.add_a_cube(pos=[0.5,0,0.04],size=[0.285,0.175,0.04],color=[0.1,0.1,0.1,1],mass=50)
        self.add_a_cube_without_collision(pos=[0.5,0,0.04],size=[0.285,0.32,0.04],color=[0.1,0.1,0.1,1])

        self.add_a_cube(pos=[-0.1,0.255,0.1],size =[0.55,0.1,0.07],color=[0.3,0.3,0.3,1],mass=500)
        self.add_a_cube(pos=[-0.1,-0.255,0.1],size=[0.55,0.1,0.07],color=[0.3,0.3,0.3,1],mass=500)
        self.wait(1)
        self.add_red_rack(centre=[0.6,0.06,0.2])
        self.add_green_rack(centre=[0.6,-0.06,0.2])        
        self.wait(1)
        self.add_red_rack(centre=[0.6,0.06,0.3])
        self.add_green_rack(centre=[0.6,-0.06,0.3])        
        self.wait(1)
        
    
    def visualize_camera_position(self):
            camPos = self._camera_pos
            pos = np.copy(camPos[:])+np.array([0,0,0.0025])
            size    = 0.05
            halfsize= size/2
            mass    = 0 #kg
            color   = [1,0,0,1]
            # lens    = p.createCollisionShape(p.GEOM_CYLINDER, radius=halfsize*2,height = 0.005 )
            # vis     = p.createVisualShape(p.GEOM_CYLINDER,radius=halfsize*2,length=0.005, rgbaColor=color, specularColor=[0,0,0])
            # obj_id  = p.createMultiBody(mass, lens, vis, pos, [0,0,0,1])
            
            ####

            color   = [0,0,0,1]
            pos     = np.copy(camPos[:])+np.array([0,0,0.025+0.0025])
            box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[halfsize, halfsize, halfsize])
            vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[halfsize, halfsize, halfsize], rgbaColor=color, specularColor=[1,1,1])
            obj_id  = p.createMultiBody(mass, box, vis, pos, [0,0,0,1])
    

    def capture_image(self,removeBackground = False): 
        bgr, depth, _ = self.camera.get_cam_img()
        
        if (removeBackground):                      
           bgr = bgr-self.bgBGRBox+self.bgBGRWithoutBox
           depth = depth-self.bgDepthBox+self.bgDepthWithoutBox

        ##convert BGR to RGB
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return bgr,depth
    def save_image(self,bgr):
        rgbim = Image.fromarray(bgr)
        # depim = Image.fromarray(bgr)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        rgbim.save('savedImages/rgbtest'+timestr+'.png')

    def creat_pile_of_tubes(self,number_of_tubes):
        obj_init_pos = [0.4, -0.]
        self.tubeObj = []

        for i in range(number_of_tubes):
            r_x    = random.uniform(obj_init_pos[0] - 0.2, obj_init_pos[0] + 0.2)
            r_y    = random.uniform(obj_init_pos[1] - 0.2, obj_init_pos[1] + 0.2)
            roll   = random.uniform(0, np.pi)
            orn    = p.getQuaternionFromEuler([roll, 0, 0])
            pos    = [r_x, r_y, 0.15]
            obj_id = p.loadURDF("objects/ycb_objects/YcbTomatoSoupCan/model.urdf", pos, orn)
            self._dummy_sim_step(50)
            self.tubeObj.append(obj_id)
            time.sleep(0.1)

        self.obj_ids = self.tubeObj    
        self._dummy_sim_step(100)

    def creat_pile_of_cube(self,number_of_cubes):
        obj_init_pos = [0.4, -0.]
        self.cube_obj = []

        for i in range(number_of_cubes):
            r_x    = random.uniform(obj_init_pos[0] - 0.2, obj_init_pos[0] + 0.2)
            r_y    = random.uniform(obj_init_pos[1] - 0.2, obj_init_pos[1] + 0.2)
            roll   = random.uniform(0, np.pi)
            orn    = p.getQuaternionFromEuler([roll, 0, 0])
            pos    = [r_x, r_y, 0.15]
            obj_id = self.add_a_cube(pos=pos,size=[0.04,0.04,0.04],color=[i/10.0,0.5,i/10.0,1])
            self._dummy_sim_step(50)
            self.cube_obj.append(obj_id)
            time.sleep(0.1)

        # self.obj_ids = self.tubeObj    
        self._dummy_sim_step(100)
        return self.cube_obj
    
    def createTempBox(self, width, no,box_centre):
        box_width = width
        box_height = 0.1
        box_z = (box_height/2)
        id1 = p.loadURDF(f'environment/urdf/objects/slab{no}.urdf',
                         [box_centre[0] - box_width /
                             2, box_centre[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id2 = p.loadURDF(f'environment/urdf/objects/slab{no}.urdf',
                         [box_centre[0] + box_width /
                             2, box_centre[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id3 = p.loadURDF(f'environment/urdf/objects/slab{no}.urdf',
                         [box_centre[0], box_centre[1] +
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        id4 = p.loadURDF(f'environment/urdf/objects/slab{no}.urdf',
                         [box_centre[0], box_centre[1] -
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        
    def create_harmony_box(self, box_centre):
        box_width = 0.29
        box_height = 0.35
        box_z = 0.2/2
        id1 = p.loadURDF(f'environment/urdf/objects/slab4.urdf',
                         [box_centre[0] - box_width / 2.0, box_centre[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id2 = p.loadURDF(f'environment/urdf/objects/slab4.urdf',
                         [box_centre[0] + box_width / 2.0, box_centre[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id3 = p.loadURDF(f'environment/urdf/objects/slab3.urdf',
                         [box_centre[0], box_centre[1] + box_height/2.0, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        id4 = p.loadURDF(f'environment/urdf/objects/slab3.urdf', 
                          [box_centre[0], box_centre[1] - box_height/2.0, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)


    def remove_drawing(self,lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)

    def visualize_predicted_grasp(self,grasps,color = [0,0,1],visibleTime =2):
       
        lineIDs = []
        for g in grasps:
            x, y, z, yaw, opening_len, obj_height = g
            opening_len = np.clip(opening_len,0,0.04)
            yaw = yaw-np.pi/2
            lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=5))
            lineIDs.append(p.addUserDebugLine([x, y, z], [x+(opening_len*np.cos(yaw)), y+(opening_len*np.sin(yaw)), z],color, lineWidth=5))
            lineIDs.append(p.addUserDebugLine([x, y, z], [x-(opening_len*np.cos(yaw)), y-(opening_len*np.sin(yaw)), z],color, lineWidth=5))
            

        self._dummy_sim_step(10)
        time.sleep(visibleTime)
        self.remove_drawing(lineIDs)

