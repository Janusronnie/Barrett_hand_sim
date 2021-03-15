#!/usr/bin/env python

import pybullet as p
import time
import pybullet_data
import numpy as np



def simulator(Initial_pos=[0, 0, 0], Initial_angle=[0, 0, 0]):
    """
    This function is used to do simulation for the gripper.
    :param Initial_pos: The initial position of the barrett hand.
                        It should be a list with three values which means at x, y, z directions.
    :param Initial_angle: The initial orientation of the barrett hand.
                        It should be a list with three values which means the rotation angle at x, y, z directions.
                        The unit of the rotation angle is the radian system.
    :return: The state of the gripper.
    """
    # connect physical server
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    # set gravity
    p.setGravity(0, 0, 0)
    # create an empty link list to store link id
    LinkId = []
    # model initial location
    StartPos = Initial_pos
    # model initial orientation in Euler
    StartOrientation = p.getQuaternionFromEuler(Initial_angle)
    # load model file and set the initial position and fixed base link
    boxId = p.loadURDF("barrett_hand_target.urdf", StartPos, StartOrientation, useFixedBase=True)
    # load object model
    object = p.loadURDF("object.urdf", useFixedBase=True)
    # set gripper be the loaded model
    gripper = boxId
    # set camera parameters
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=0, cameraTargetPosition=[0.5, 0.5, 0.2])

    # for loop to obtain the joint information and set parameters
    for i in range(0, p.getNumJoints(gripper)):
        p.setJointMotorControl2(gripper, i, p.POSITION_CONTROL, targetPosition=0, force=0)
        # obtain the limit rotation angle range of the joint
        lower_limit = p.getJointInfo(gripper, i)[8]
        upper_limit = p.getJointInfo(gripper, i)[9]
        # obtain the joint name
        linkName = p.getJointInfo(gripper, i)[12].decode("ascii")
        # set the gui control board
        LinkId.append(p.addUserDebugParameter(linkName, lower_limit, upper_limit, 0))

    while p.isConnected():
        # start do simulation
        p.stepSimulation()
        time.sleep(1. / 240.)

        # move joints following command
        for i in range(0, p.getNumJoints(gripper)):
            linkPos = p.readUserDebugParameter((LinkId[i]))
            p.setJointMotorControl2(gripper, i, p.POSITION_CONTROL, targetPosition=linkPos)
        p.saveBullet('Test.bullet')
    p.disconnect()


if __name__ == "__main__":
    simulator(Initial_pos=[0.2, 0, 0], Initial_angle=[0, 0, 0])

