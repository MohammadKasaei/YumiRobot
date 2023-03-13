import time
import numpy as np
import pybullet as p
from environment.camera.camera import Camera, CameraIntrinsic
import pybullet_data
from collections import namedtuple
from operator import methodcaller


class YumiEnv():
    def __init__(self) -> None:
        self.simulationStepTime = 0.005
        self.vis = True

        p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.simulationStepTime)
        # sim_arg = {'gui':'True'}
        self.plane_id = p.loadURDF('plane.urdf')
        self.load_robot(urdf = 'urdfs/yumi_grippers.urdf',print_joint_info=True)
        self.go_home()        
        self.move_left_gripper (gw=0)
        self.move_right_gripper (gw=0)
        self._dummy_sim_step(1000)

        print("\n\n\nRobot is armed and ready to use...\n\n\n")
        print ('-'*40)

        
        camera_pos = np.array([0.3, -0., 0.6])
        camera_target = np.array([camera_pos[0], camera_pos[1], 0.0])        
        self._init_camera(camera_pos,camera_target)

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
        
        self._left_ee_frame_name  = 'yumi_joint_6_l'
        self._right_ee_frame_name = 'yumi_joint_6_r'
        
        self._LEFT_HOME_POSITION = [-0.473, -1.450, 1.091, 0.031, 0.513, 0.77, -1.669]
        self._RIGHT_HOME_POSITION = [0.413, -1.325, -1.040, -0.053, -0.484, 0.841, -1.546]

        self._RIGHT_HAND_JOINT_IDS = [1,2,3,4,5,6,7]
        self._RIGHT_GRIP_JOINT_IDS = [9,10]
                
        self._LEFT_HAND_JOINT_IDS = [11,12,13,14,15,16,17]
        self._LEFT_GRIP_JOINT_IDS = [19,20]

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

            if info.type == "REVOLUTE" or info.type == "PRISMATIC":  # set revolute joint to static
                p.setJointMotorControl2(self.robot_id, info.id, p.POSITION_CONTROL, targetPosition=0, force=0)
                if print_joint_info:
                    print (info)
                    print (jointType)                            
                    print ('-'*40)

    def _init_camera(self,camera_pos,camera_target,visulize_camera = True):        
        self._camera_pos = camera_pos
        self._camera_target = camera_target
        IMG_SIZE = 220
        self.camera = Camera(cam_pos=self._camera_pos, cam_target= self._camera_target, near = 0.2, far = 2, size= [IMG_SIZE, IMG_SIZE], fov=60)
        if visulize_camera:
            self.visualize_camera_position(self._camera_pos)



    def _dummy_sim_step(self,n):
        for _ in range(n):
            p.stepSimulation()

    

    def go_home(self):
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._LEFT_HAND_JOINT_IDS,targetPositions  = self._LEFT_HOME_POSITION)
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._RIGHT_HAND_JOINT_IDS,targetPositions = self._RIGHT_HOME_POSITION)
        self._dummy_sim_step(100)

    def get_left_ee_state(self):
        return p.getLinkState(self.robot_id,self._left_ee_frame_name)
    
    def get_right_ee_state(self):
        return p.getLinkState(self.robot_id,self._right_ee_frame_name)

    def move_left_arm(self,pose):                
        joint_poses = p.calculateInverseKinematics(self.robot_id, self._LEFT_HAND_JOINT_IDS[-1], pose[0], pose[1])
        joint_poses = list(map(self._ang_in_mpi_ppi, joint_poses))
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._LEFT_HAND_JOINT_IDS,targetPositions  = joint_poses[9:16])

        
    
    def move_right_arm(self,pose):        
        joint_poses = p.calculateInverseKinematics(self.robot_id, self._RIGHT_HAND_JOINT_IDS[-1], pose[0], pose[1])
        joint_poses = list(map(self._ang_in_mpi_ppi, joint_poses))

        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._RIGHT_HAND_JOINT_IDS,targetPositions  = joint_poses[:7], )
    
    
    def move_left_gripper(self, gw=0):
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._LEFT_GRIP_JOINT_IDS,targetPositions = [gw,gw])
        
    def move_right_gripper(self, gw=0):
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._RIGHT_GRIP_JOINT_IDS,targetPositions = [gw,gw])

  
    def add_a_cube(self,pos,size=[0.1,0.1,0.1],mass = 0.1, color = [1,1,0,1]):

        cubesID = []
        box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = p.createMultiBody(mass, box, vis, pos, [0,0,0,1])
        p.changeDynamics(obj_id, 
                        -1,
                        spinningFriction=0.001,
                        rollingFriction=0.001,
                        linearDamping=0.0)
        cubesID.append(obj_id)
        p.stepSimulation()
    
    

    def visualize_camera_position(self,camPos):
            pos = np.copy(camPos[:])+np.array([0,0,0.0025])
            size    = 0.05
            halfsize= size/2
            mass    = 0 #kg
            color   = [1,0,0,1]
            lens    = p.createCollisionShape(p.GEOM_CYLINDER, radius=halfsize*2,height = 0.005 )
            vis     = p.createVisualShape(p.GEOM_CYLINDER,radius=halfsize*2,length=0.005, rgbaColor=color, specularColor=[0,0,0])
            obj_id  = p.createMultiBody(mass, lens, vis, pos, [0,0,0,1])
            
            ####

            color   = [0,0,0,1]
            pos     = np.copy(camPos[:])+np.array([0,0,0.025+0.0025])
            box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[halfsize, halfsize, halfsize])
            vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[halfsize, halfsize, halfsize], rgbaColor=color, specularColor=[1,1,1])
            obj_id  = p.createMultiBody(mass, box, vis, pos, [0,0,0,1])

        

if __name__ == '__main__':

    env = YumiEnv()


    gw = i = 0 

    while (True):
        gw = 1 if gw == 0 else 0
        

        env.add_a_cube([0.5,0.,0.18],size=[0.04,0.04,0.04])

        env.move_left_gripper (gw=gw)
        env.move_right_gripper (gw=gw)
        
        env._dummy_sim_step(1)


        for i in range(100):    
            if i % 10 == 0:
                env.camera.get_cam_img()
            ori = p.getQuaternionFromEuler([0,np.pi,0])
            pose = [[0.3,0.4-(i*0.006),0.2],ori]            
            env.move_left_arm(pose=pose)
            env._dummy_sim_step(50)
            # time.sleep(0.01)





        # env.move_left_arm(pose=pose)
        
        time.sleep(0.1)
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # p.stepSimulation()
        # time.sleep(0.010)

