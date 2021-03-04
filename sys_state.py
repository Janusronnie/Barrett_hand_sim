import random
import pybullet as p
import time
import pybullet_data
import numpy as np
from cal_dist_sdf import cal_dist


def transform_matrix(trans=[0, 0, 0], rot=[0, 0, 0]):
    M = np.eye(4, 4)

    M[0, 3] = trans[0]
    M[1, 3] = trans[1]
    M[2, 3] = trans[2]

    R_x = np.eye(3, 3)
    R_x[0, :] = [1, 0, 0]
    R_x[1, :] = [0, np.cos(rot[0]), -np.sin(rot[0])]
    R_x[2, :] = [0, np.sin(rot[0]), np.cos(rot[0])]

    R_y = np.eye(3, 3)
    R_y[0, :] = [np.cos(rot[1]), 0, np.sin(rot[1])]
    R_y[1, :] = [0, 1, 0]
    R_y[2, :] = [-np.sin(rot[1]), 0, np.cos(rot[1])]

    R_z = np.eye(3, 3)
    R_z[0, :] = [np.cos(rot[2]), -np.sin(rot[2]), 0]
    R_z[1, :] = [np.sin(rot[2]), np.cos(rot[2]), 0]
    R_z[2, :] = [0, 0, 1]

    R = R_z @ R_y @ R_x

    M[:3, :3] = R

    return M


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

        self.Aw = 0.025
        self.A1 = 0.05
        self.A2 = 0.07
        self.A3 = 0.05
        self.Dw = 0.084
        self.D3 = 0.0095

    def target(self, pos, ori):
        self.base_pos = pos  # [0, 1]
        self.base_ori = ori  # [-3.14, 3.14]

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

    def palm_contact_matrix(self):
        P_Link0_1 = transform_matrix(trans=[0, 0, self.Dw])
        palm_mtx = P_Link0_1
        return palm_mtx

    def finger1_contact_matrix(self):
        F1_Link0_1 = transform_matrix(trans=[0, 0, self.Dw])
        F1_Link1_2 = transform_matrix(trans=[0.5 * self.D3, 0, 0])
        F1_Link2_3 = transform_matrix(trans=[0, self.A1, 0], rot=[0, 0, self.finger1_prox])
        F1_Link3_4 = transform_matrix(trans=[0, 0.5 * self.A2, self.D3], rot=[self.finger1_med, 0, 0])
        F1_Link3_med = transform_matrix(trans=[0, 0.5 * self.A2, 0], rot=[self.finger1_med, 0, 0])

        finger1_med_start = F1_Link0_1 @ F1_Link1_2 @ F1_Link2_3 @ F1_Link3_med
        finger1_med_mtx = F1_Link0_1 @ F1_Link1_2 @ F1_Link2_3 @ F1_Link3_4

        F1_Link3_5 = transform_matrix(trans=[0, self.A2, 0], rot=[self.finger1_med, 0, 0])
        F1_Link5_6 = transform_matrix(trans=[0, 0.5 * self.A3 * np.cos(42 / 180 * np.pi),
                                          0.5 * self.A3 * np.sin(42 / 180 * np.pi)], rot=[self.finger1_dist, 0, 0])
        F1_Link6_7 = transform_matrix(trans=[0, - self.D3 * np.sin(42 / 180 * np.pi), self.D3 * np.cos(42 / 180 * np.pi)])

        finger1_dist_start = F1_Link0_1 @ F1_Link1_2 @ F1_Link2_3 @ F1_Link3_5 @ F1_Link5_6
        finger1_dist_mtx = F1_Link0_1 @ F1_Link1_2 @ F1_Link2_3 @ F1_Link3_5 @ F1_Link5_6 @ F1_Link6_7

        return finger1_med_mtx, finger1_dist_mtx, finger1_med_start, finger1_dist_start

    def finger2_contact_matrix(self):
        F2_Link0_1 = transform_matrix(trans=[0, 0, self.Dw])
        F2_Link1_2 = transform_matrix(trans=[- 0.5 * self.D3, 0, 0])
        F2_Link2_3 = transform_matrix(trans=[0, self.A1, 0], rot=[0, 0, self.finger2_prox])
        F2_Link3_4 = transform_matrix(trans=[0, 0.5 * self.A2, self.D3], rot=[self.finger2_med, 0, 0])
        F2_Link3_med = transform_matrix(trans=[0, 0.5 * self.A2, 0], rot=[self.finger1_med, 0, 0])

        finger2_med_start = F2_Link0_1 @ F2_Link1_2 @ F2_Link2_3 @ F2_Link3_med
        finger2_med_mtx = F2_Link0_1 @ F2_Link1_2 @ F2_Link2_3 @ F2_Link3_4

        F2_Link3_5 = transform_matrix(trans=[0, self.A2, 0], rot=[self.finger2_med, 0, 0])
        F2_Link5_6 = transform_matrix(trans=[0, 0.5 * self.A3 * np.cos(42 / 180 * np.pi),
                                          0.5 * self.A3 * np.sin(42 / 180 * np.pi)], rot=[self.finger2_dist, 0, 0])
        F2_Link6_7 = transform_matrix(trans=[0, - self.D3 * np.sin(42 / 180 * np.pi), self.D3 * np.cos(42 / 180 * np.pi)])

        finger2_dist_start = F2_Link0_1 @ F2_Link1_2 @ F2_Link2_3 @ F2_Link3_5 @ F2_Link5_6
        finger2_dist_mtx = F2_Link0_1 @ F2_Link1_2 @ F2_Link2_3 @ F2_Link3_5 @ F2_Link5_6 @ F2_Link6_7

        return finger2_med_mtx, finger2_dist_mtx, finger2_med_start, finger2_dist_start

    def finger3_contact_matrix(self):
        F3_Link0_1 = transform_matrix(trans=[0, 0, self.Dw])
        F3_Link1_2 = transform_matrix(trans=[0, -self.A1, 0])
        F3_Link2_3 = transform_matrix(trans=[0, -0.5 * self.A2, self.D3], rot=[self.finger3_med, 0, 0])
        F3_Link2_med = transform_matrix(trans=[0, -0.5 * self.A2, 0], rot=[self.finger3_med, 0, 0])

        finger3_med_start = F3_Link0_1 @ F3_Link1_2 @ F3_Link2_med
        finger3_med_mtx = F3_Link0_1 @ F3_Link1_2 @ F3_Link2_3

        F3_Link2_4 = transform_matrix(trans=[0, -self.A2, 0], rot=[self.finger3_med, 0, 0])
        F3_Link4_5 = transform_matrix(trans=[0, -0.5 * self.A3 * np.cos(42 / 180 * np.pi),
                                          0.5 * self.A3 * np.sin(42 / 180 * np.pi)], rot=[self.finger3_dist, 0, 0])
        F3_Link5_6 = transform_matrix(trans=[0, self.D3 * np.sin(42 / 180 * np.pi), self.D3 * np.cos(42 / 180 * np.pi)])

        finger3_dist_start = F3_Link0_1 @ F3_Link1_2 @ F3_Link2_4 @ F3_Link4_5
        finger3_dist_mtx = F3_Link0_1 @ F3_Link1_2 @ F3_Link2_4 @ F3_Link4_5 @ F3_Link5_6

        return finger3_med_mtx, finger3_dist_mtx, finger3_med_start, finger3_dist_start

    def contact_pts_and_n_vector(self):
        trans_mtx_hand_to_world = transform_matrix(trans=self.base_pos, rot=self.base_ori)
        base = np.ones([4, 1])
        base[0, 0] = self.base_pos[0]
        base[1, 0] = self.base_pos[1]
        base[2, 0] = self.base_pos[2]

        palm_c = list((trans_mtx_hand_to_world @ self.palm_contact_matrix() @ base)[:3, 0])
        finger1_med_c = list((trans_mtx_hand_to_world @ self.finger1_contact_matrix()[0] @ base)[:3, 0])
        finger1_dist_c = list((trans_mtx_hand_to_world @ self.finger1_contact_matrix()[1] @ base)[:3, 0])
        finger2_med_c = list((trans_mtx_hand_to_world @ self.finger2_contact_matrix()[0] @ base)[:3, 0])
        finger2_dist_c = list((trans_mtx_hand_to_world @ self.finger2_contact_matrix()[1] @ base)[:3, 0])
        finger3_med_c = list((trans_mtx_hand_to_world @ self.finger3_contact_matrix()[0] @ base)[:3, 0])
        finger3_dist_c = list((trans_mtx_hand_to_world @ self.finger3_contact_matrix()[1] @ base)[:3, 0])

        contact_pts = {'palm_p': palm_c, 'finger1_med_p': finger1_med_c, 'finger1_dist_p': finger1_dist_c,
                           'finger2_med_p': finger2_med_c, 'finger2_dist_p': finger2_dist_c,
                           'finger3_med_p': finger3_med_c, 'finger3_dist_p': finger3_dist_c}

        palm_n_start_pts = base[:3, 0]
        finger1_med_n_start_pts = (trans_mtx_hand_to_world @ self.finger1_contact_matrix()[2] @ base)[:3, 0]
        finger1_dist_n_start_pts = (trans_mtx_hand_to_world @ self.finger1_contact_matrix()[3] @ base)[:3, 0]
        finger2_med_n_start_pts = (trans_mtx_hand_to_world @ self.finger2_contact_matrix()[2] @ base)[:3, 0]
        finger2_dist_n_start_pts = (trans_mtx_hand_to_world @ self.finger2_contact_matrix()[3] @ base)[:3, 0]
        finger3_med_n_start_pts = (trans_mtx_hand_to_world @ self.finger3_contact_matrix()[2] @ base)[:3, 0]
        finger3_dist_n_start_pts = (trans_mtx_hand_to_world @ self.finger3_contact_matrix()[3] @ base)[:3, 0]

        palm_n = list(np.array(palm_c) - palm_n_start_pts)
        finger1_med_n = list(np.array(finger1_med_c) - finger1_med_n_start_pts)
        finger1_dist_n = list(np.array(finger1_dist_c) - finger1_dist_n_start_pts)
        finger2_med_n = list(np.array(finger2_med_c) - finger2_med_n_start_pts)
        finger2_dist_n = list(np.array(finger2_dist_c) - finger2_dist_n_start_pts)
        finger3_med_n = list(np.array(finger3_med_c) - finger3_med_n_start_pts)
        finger3_dist_n = list(np.array(finger3_dist_c) - finger3_dist_n_start_pts)

        contact_n_vector = {'palm_n': palm_n, 'finger1_med_n': finger1_med_n, 'finger1_dist_n': finger1_dist_n,
                           'finger2_med_n': finger2_med_n, 'finger2_dist_n': finger2_dist_n,
                           'finger3_med_n': finger3_med_n, 'finger3_dist_n': finger3_dist_n}

        return contact_pts, contact_n_vector


def cal_distance(pts):
    dist = []
    trans_world_to_object = transform_matrix(trans=[-0.5, -0.5, -0.1])
    for keys in pts.keys():
        pts_array = np.ones([4, 1])
        pts_array[:3, 0] = np.array(pts[keys]).T
        contact_to_obj_pts = list((trans_world_to_object @ pts_array)[:3, 0])
        dist.append(cal_dist(point_loc=contact_to_obj_pts))
    return dist


if __name__ == "__main__":
    state = system_state()
    # state.random_state()
    state.target(pos=[0, 0, 0], ori=[0, 0, 0])
    contact_pts, contact_n_vector = state.contact_pts_and_n_vector()
    dist = cal_distance(pts=contact_pts)
    # for keys in contact_pts.keys():
    #     print('The location of {0} is {1}'.format(keys, contact_pts[keys]))
    # print('\n')
    # for keys in contact_n_vector.keys():
    #     print('The normal vector of {0} is {1}'.format(keys, contact_n_vector[keys]))
    # state.simulator()
