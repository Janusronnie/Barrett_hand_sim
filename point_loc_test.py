import numpy as np


class sys_loc:

    def __init__(self, loc, ori):
        self.Aw = 0.025
        self.A1 = 0.05
        self.A2 = 0.07
        self.A3 = 0.05
        self.Dw = 0.084
        self.D3 = 0.0095

        self.ori_loc = loc
        self.ori_ang = ori

    def transform_matrix(self):
        M = np.eye(4, 4)
        T = np.zeros(3, 1)
        R_x = np.eye(3, 3)
        R_x = np.array([1, 0, 0], [0, np.cos(self.ori_ang[0]), -np.sin(self.ori_ang[0])],
                       [0, np.sin(self.ori_ang[0]), np.cos(self.ori_ang[0])])
        A = 1

if __name__ == "__main__":
    hand = sys_loc(loc=[0, 0, 0], ori=[0, 0, 0])
    hand.transform_matrix()