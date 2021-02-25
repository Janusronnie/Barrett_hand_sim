import random
import pybullet as p
import time
import pybullet_data


class system_state:

    def __init__(self):
        self.base_pos = [0, 0, 0]  # [0, 1]
        self.base_ori = [0, 0, 0]  # [-3.14, 3.14]
        self.finger1_prox = 0  # [0, 3.14]
        self.finger1_med = 0  # [0, 2.442]
        self.finger1_dist = 0  # [0, 0.837]
        self.finger2_prox = 0  # [-3.14, 0]
        self.finger2_med = 0  # [0, 2.442]
        self.finger2_dist = 0  # [0, 0.837]
        self.finger3_med = 0  # [-2.442, 0]
        self.finger3_dist = 0  # [0, 0.837]
        self.angle_state = [self.finger1_prox, self.finger1_med, self.finger1_dist, self.finger2_prox,
                            self.finger2_med, self.finger2_dist, self.finger3_med, self.finger3_dist]

    def random_state(self):
        base_pos_x = random.uniform(0, 1)
        base_pos_y = random.uniform(0, 1)
        base_pos_z = random.uniform(0, 1)
        self.base_pos = [base_pos_x, base_pos_y, base_pos_z]
        base_ori_x = random.uniform(-3.14, 3.14)
        base_ori_y = random.uniform(-3.14, 3.14)
        base_ori_z = random.uniform(-3.14, 3.14)
        self.base_ori = [base_ori_x, base_ori_y, base_ori_z]
        self.finger1_prox = random.uniform(0, 3.14)
        self.finger1_med = random.uniform(0, 2.442)
        self.finger1_dist = random.uniform(0, 0.837)
        self.finger2_prox = random.uniform(-3.14, 0)
        self.finger2_med = random.uniform(0, 2.442)
        self.finger2_dist = random.uniform(0, 0.837)
        self.finger3_med = random.uniform(-2.442, 0)
        self.finger3_dist = random.uniform(0, 0.837)
        self.angle_state = [self.finger1_prox, self.finger1_med, self.finger1_dist, self.finger2_prox,
                            self.finger2_med, self.finger2_dist, self.finger3_med, self.finger3_dist]

    def simulator(self):
        """
        This function is used to do simulation for the gripper.
        :param Initial_pos: The initial position of the barrett hand.
                            It should be a list with three values which means at x, y, z directions.
        :param Initial_angle: The initial orientation of the barrett hand.
                            It should be a list with three values which means the rotation angle at x, y, z directions.
                            The unit of the rotation angle is the radian system.
        :return: The state of the gripper.
        """
        # connect gpu
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        # set gravity
        p.setGravity(0, 0, 0)
        # model initial location
        StartPos = self.base_pos
        # model initial orientation in Euler
        StartOrientation = p.getQuaternionFromEuler(self.base_ori)
        # load model file and set the initial position and fixed base link
        boxId = p.loadURDF("barrett_hand.urdf", StartPos, StartOrientation, useFixedBase=True)
        # load object model
        object = p.loadURDF("object.urdf", useFixedBase=True)
        # set gripper be the loaded model
        gripper = boxId
        # set camera parameters
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=0, cameraTargetPosition=[0.5, 0.5, 0.2])

        while p.isConnected():
            # start do simulation
            p.stepSimulation()
            time.sleep(1. / 240.)

            # move joints following command
            for i in range(0, p.getNumJoints(gripper)):
                linkPos = self.angle_state[i]
                p.setJointMotorControl2(gripper, i, p.POSITION_CONTROL, targetPosition=linkPos)

        p.disconnect()


if __name__ == "__main__":
    state = system_state()
    state.random_state()
    state.simulator()