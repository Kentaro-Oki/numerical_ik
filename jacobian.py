import numpy as np
from fk import FK

class Jacobian():
    def __init__(self):
        self.fk = FK()
        self.EPSILON = 1e-8

    # Error of two rotation matrices (from rm1 to rm2), expressed as rotation vector
    def error_rms(self, rm1, rm2):
        return self.fk.rm2rv(rm2 @ rm1.T)

    def jacob_pvrm_mat(self, theta, fk_pvrm):
        self._l_theta = len(theta)
        self.J_mat = np.zeros((6,self._l_theta))
        for i in range(self._l_theta):
            self._theta_p = theta + self.EPSILON/2*np.identity(self._l_theta)[i,:]
            self._theta_m = theta - self.EPSILON/2*np.identity(self._l_theta)[i,:]
            self.J_mat[:3,i] = (fk_pvrm(self._theta_p)[0] - fk_pvrm(self._theta_m)[0])/self.EPSILON
            self.J_mat[3:,i] = self.error_rms(fk_pvrm(self._theta_m)[1], fk_pvrm(self._theta_p)[1])/self.EPSILON
        return self.J_mat

    def jacob_pv_mat(self, theta, fk_pvrm):
        self._l_theta = len(theta)
        self.J_mat = np.zeros((3,self._l_theta))
        for i in range(self._l_theta):
            self._theta_p = theta + self.EPSILON/2*np.identity(self._l_theta)[i,:]
            self._theta_m = theta - self.EPSILON/2*np.identity(self._l_theta)[i,:]
            self.J_mat[:,i] = (fk_pvrm(self._theta_p)[0] - fk_pvrm(self._theta_m)[0])/self.EPSILON
        return self.J_mat

    def jacob_pvs_mat(self, theta, fk_pvs):
        self._l_theta = len(theta)
        self._l_pvs = len(fk_pvs(theta))
        self.J_mat = np.zeros((self._l_pvs, self._l_theta))
        for i in range(self._l_theta):
            self._theta_p = theta + self.EPSILON/2*np.identity(self._l_theta)[i,:]
            self._theta_m = theta - self.EPSILON/2*np.identity(self._l_theta)[i,:]
            self.J_mat[:,i] = (fk_pvs(self._theta_p) - fk_pvs(self._theta_m))/self.EPSILON
        return self.J_mat

    def arm(self, theta):
        return self.jacob_pvrm_mat(theta, self.fk.fk_arm_right_pvrm)

    def elbow(self, theta):
        return self.jacob_pv_mat(theta, self.fk.fk_elbow_right_pvrm)

    def wrist(self, theta):
        return self.jacob_pv_mat(theta, self.fk.fk_wrist_right_pvrm)

    def finger_tips(self, theta):
        return self.jacob_pvs_mat(theta, self.fk.fk_fingers_right_pv)