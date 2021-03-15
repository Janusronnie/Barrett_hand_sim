import sys_state
import pybullet as p
import time
import pybullet_data
import numpy as np
import math
import matplotlib.pyplot as plt


def quality_function(vector_dict, normal):
    """
    This function is used to calculate the grasping quality score
    :param vector_dict: The shortest distance vector
    :param normal: The normal vector from the contact point
    :return: The grasping quality score
    """
    delta = 0
    for item in vector_dict.keys():
        O = vector_dict[item]['distance']
        N = normal[item]
        delta += np.linalg.norm(O) + (1 - np.dot(N, O) / np.linalg.norm(O))
    Q = (1 - delta)
    return Q


def jump(delta_E, Tem):
    """
    This function is used to return the result of jumping to new state
    :param delta_E: The difference of the old Q value and the new Q value
    :param Tem: The current temperature
    :return: True or False
    """
    # if the new state is better than the old state
    if delta_E < 0:
        return True
    # if the new state is worse than the old state
    else:
        # calculate the current probability
        d = math.exp(-delta_E / Tem)
        if d > np.random.rand():
            return True
        else:
            return False


def simulator(Initial_pos=[0, 0, 0], Initial_angle=[0, 0, 0]):
    """
    This function is used to do simulation
    :param Initial_pos: The initial position of the base link
    :param Initial_angle: The initial orientation of the base link
    """
    # initial the robot class
    robot = sys_state.system_state()
    # set the starting values
    T = 1e5
    alpha = 0.99
    old_Q = -20
    iteration = 0
    max_iteration = 20000
    Q_list = []

    # connect physical server
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    # set gravity
    p.setGravity(0, 0, 0)
    # load model file and set the initial position and fixed base link
    gripper = p.loadURDF("barrett_hand_target.urdf", useFixedBase=True)
    # load object model
    object = p.loadURDF("object.urdf", useFixedBase=True)
    # set camera parameters
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=0, cameraTargetPosition=[0.5, 0.5, 0.2])

    while iteration != max_iteration:
        print("Iteration Number is ", iteration)
        # start do simulation
        p.stepSimulation()
        time.sleep(1. / 240.)

        # move joints to the new state
        moveable_joint_ID = [0, 1, 2, 5, 6, 7, 10, 11]
        for i in range(len(moveable_joint_ID)):
            linkPos = robot.angle_state[i]
            p.setJointMotorControl2(gripper, moveable_joint_ID[i], p.POSITION_CONTROL, targetPosition=linkPos)

        # create a dictionary to store values
        vector = {'f1_dist': {'gripper': None, 'distance': None, 'normal': None},
                  'f1_med': {'gripper': None, 'distance': None, 'normal': None},
                  'f2_dist': {'gripper': None, 'distance': None, 'normal': None},
                  'f2_med': {'gripper': None, 'distance': None, 'normal': None},
                  'f3_dist': {'gripper': None, 'distance': None, 'normal': None},
                  'f3_med': {'gripper': None, 'distance': None, 'normal': None},
                  'base': {'gripper': None, 'distance': None, 'normal': None}}

        # store the contact point location, the distance vector and the normal vector
        contact_joint_linkID = [3, 4, 8, 9, 12, 13, 14]

        key = list(vector.keys())
        for n in range(len(contact_joint_linkID)):
            # the contact point location
            gripper_c = np.array(p.getClosestPoints(gripper, object, 5.0, linkIndexA=contact_joint_linkID[n])[0][5])
            vector[key[n]]['gripper'] = list(gripper_c)
            # the distance vector
            object_c = np.array(p.getClosestPoints(gripper, object, 5.0, linkIndexA=contact_joint_linkID[n])[0][6])
            vector[key[n]]['distance'] = object_c - gripper_c
            # the normal vector
            vector[key[n]]['normal'] = \
                -np.array(p.getClosestPoints(gripper, object, 5.0, linkIndexA=contact_joint_linkID[n])[0][7])

        # collision detection
        distance = []
        for item in vector.keys():
            distance.append(np.linalg.norm(vector[item]['distance']))
        # if the collision does not happen
        if (np.array(distance)).all() > 0.025:
            # calculate the quality score of the new state
            new_Q = quality_function(vector_dict=vector, normal=robot.contact_pts_and_n_vector())
            # store the quality value
            Q_list.append(old_Q)
            # calculate the difference between the quality scores of the two states
            delta_Q = old_Q - new_Q
            # judge whether jump to new state
            if jump(delta_E=delta_Q, Tem=T):
                old_Q = new_Q
                # save the current state
                p.saveBullet('result_state/Best.bullet')
                T = alpha * T
            # update the iteration number
            iteration += 1
        # generate the new random state
        robot.new_random_state()
        # set the base link to the new state
        p.resetBasePositionAndOrientation(gripper, robot.base_pos, p.getQuaternionFromEuler(robot.base_ori))
    p.disconnect()

    # plot the quality versus iteration image
    plt.plot(Q_list, 'k-')
    plt.xlabel('Iteration Number')
    plt.ylabel('Quality')
    plt.grid()
    plt.savefig('result_plot/Quality plot for {0} iterations.png'.format(max_iteration))
    plt.show()


if __name__ == "__main__":
    # run the simulator function
    simulator(Initial_pos=[0, 0, 0], Initial_angle=[0, 0, 0])

    # load the best simulation state
    physicsClient = p.connect(p.GUI)
    boxId = p.loadURDF("barrett_hand_target.urdf", useFixedBase=True)
    object = p.loadURDF("object.urdf", useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=0, cameraTargetPosition=[0.5, 0.5, 0.2])
    while p.isConnected():
        p.stepSimulation()
        p.restoreState(fileName='result_state/saved.bullet')