import numpy as np

class DH_param():
    def __init__(self):
        self.LINK_LEN_ARM_RIGHT = np.array([0.1, 0.6, 0.315, 0., 0.33, 0.02, 0.428, 0.0045, 0.03])
        self.LINK_LEN_THUMB_RIGHT = np.array([78e-3, 15e-3, 6.7e-3, 36e-3, 4.5e-3, 27.5e-3, 40e-3, 20e-3, 20e-3])
        self.LINK_LEN_INDEX_RIGHT = np.array([156.5e-3, 17.5e-3, 6.2e-3, 20e-3, 3e-3, 45.5e-3, 20e-3, 20e-3])
        self.LINK_LEN_MIDDLE_RIGHT = np.array([161.5e-3, 17.5e-3, 18.3e-3, 20e-3, 3e-3, 50.5e-3, 20e-3, 20e-3])

    def arm(self, theta, link_param):
        return np.array([[link_param[0], 0., 0., -np.pi/2],
                        [link_param[1], link_param[2], 0., theta[0] + np.pi/2],
                        [0., link_param[3], -np.pi/2, theta[1]],
                        [0., link_param[4], np.pi/2, theta[2]],
                        [link_param[5], 0., -np.pi/2, theta[3]],
                        [0., link_param[6], np.pi/2, theta[4] + np.pi/2],
                        [link_param[7], 0., np.pi/2, theta[5] + np.pi/2],
                        [link_param[8], 0., np.pi/2, theta[6]],
                        [0., 0, -np.pi/2, np.pi]])

    def arm_right(self, theta):
        return self.arm(theta, self.LINK_LEN_ARM_RIGHT)

    def thumb_right(self, theta):
        return np.array([[-self.LINK_LEN_THUMB_RIGHT[0], self.LINK_LEN_THUMB_RIGHT[1], 0., np.pi/2],
                        [self.LINK_LEN_THUMB_RIGHT[2], 0., np.pi/2, theta[7]],
                        [0., self.LINK_LEN_THUMB_RIGHT[3], -np.pi/2, theta[8] + np.pi/2],
                        [self.LINK_LEN_THUMB_RIGHT[4], 0., -np.pi/2, theta[9]],
                        [self.LINK_LEN_THUMB_RIGHT[5], 0., 0., -np.pi/2],
                        [self.LINK_LEN_THUMB_RIGHT[6], 0., 0., theta[10] + np.pi/2],
                        [self.LINK_LEN_THUMB_RIGHT[7], self.LINK_LEN_THUMB_RIGHT[8], np.pi/2, 0.]])

    def index_right(self, theta):
        return np.array([[-self.LINK_LEN_INDEX_RIGHT[0], self.LINK_LEN_INDEX_RIGHT[1], 0., np.pi/2],
                        [self.LINK_LEN_INDEX_RIGHT[2], 0., np.pi, theta[11] + np.pi/2],
                        [-self.LINK_LEN_INDEX_RIGHT[3], 0., np.pi/2, theta[12] - np.pi/2],
                        [self.LINK_LEN_INDEX_RIGHT[4], 0., 0., -np.pi/2],
                        [self.LINK_LEN_INDEX_RIGHT[5], 0., 0., theta[13] + np.pi/2],
                        [self.LINK_LEN_INDEX_RIGHT[6], self.LINK_LEN_INDEX_RIGHT[7], np.pi/2, 0.]])

    def middle_right(self, theta):
        return np.array([[-self.LINK_LEN_MIDDLE_RIGHT[0], self.LINK_LEN_MIDDLE_RIGHT[1], 0., np.pi/2],
                        [-self.LINK_LEN_MIDDLE_RIGHT[2], 0., np.pi, theta[14] + np.pi/2],
                        [-self.LINK_LEN_MIDDLE_RIGHT[3], 0., np.pi/2, theta[15] - np.pi/2],
                        [self.LINK_LEN_MIDDLE_RIGHT[4], 0., 0., -np.pi/2],
                        [self.LINK_LEN_MIDDLE_RIGHT[5], 0., 0., theta[16] + np.pi/2],
                        [self.LINK_LEN_MIDDLE_RIGHT[6], self.LINK_LEN_MIDDLE_RIGHT[7], np.pi/2, 0.]])


