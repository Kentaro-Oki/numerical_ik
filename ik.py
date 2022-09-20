import numpy as np
from scipy.spatial.transform import Rotation as R
from fk import FK
from jacobian import Jacobian

class IK:
    def __init__(self):
        self.POS_EPSILON = 1e-4
        self.ROT_EPSILON = 1e-6
        self.DAMP_PARAM = np.concatenate((1.1e-3**np.ones(7), 0.1*1e-3*np.ones(4), \
                                        0.1*1e-3*np.ones(3), 0.1*1e-3*np.ones(3))) # Damp for arm, thumb, index and middle joints
        self.JOINT_LIMIT_MIN = np.radians(np.array([-190., -90., -360., -15., -360., -60., -70., \
                                                    -90., -90., 0., -45., \
                                                    -25., 0., -20., \
                                                    -90., 0., -20.]))
        self.JOINT_LIMIT_MAX = np.radians(np.array([190., 90., 360., 180., 360., 60., 70., \
                                                    30., 60., 100., 100., \
                                                    90., 100., 140., \
                                                    25., 100., 140.]))

        self.fk = FK()
        self.jacob = Jacobian()

    # Calc. error of position vector and rotation matrix from theta of current pose and target position vector and rotation matrix
    def error_pvrm(self, theta, tgt_pvrm, fk_pvrm):
        self.e_pv = tgt_pvrm[0] - fk_pvrm(theta)[0]
        self.e_rv = self.jacob.error_rms(fk_pvrm(theta)[1], tgt_pvrm[1])
        return np.concatenate((self.e_pv, self.e_rv))

    # Calc. error of position vector and rotation matrix from theta of current pose and target position vector and rotation matrix
    def error_pvs(self, theta, tgt_pvs, fk_pvs):
        self.e_pvs = tgt_pvs[:3] - fk_pvs(theta)[:3]
        for i in range(int(len(tgt_pvs)/3)-1):
            self._tmp_err = tgt_pvs[3*(i+1):3*(i+2)] - fk_pvs(theta)[3*(i+1):3*(i+2)]
            self.e_pvs = np.append(self.e_pvs, self._tmp_err)
        return self.e_pvs

    # Simple numerical IK for a target set of position vector and rotation matrix from initial theta 
    def ik_pvrm_simple(self, init_theta, tgt_pvrm, fk_pvrm, jacob_pvrm, is_jrange_mode):
        self._W_E_mat = np.diag(np.ones(6))
        self._W_N_bar_mat = np.diag(self.DAMP_PARAM)
        self.theta = init_theta
        self._prev_e_vec = np.zeros(6)
        while(True):
            self._e_vec = self.error_pvrm(self.theta, tgt_pvrm, fk_pvrm).reshape(-1,1)
            if (np.linalg.norm(self._e_vec[:,0][:3]) < self.POS_EPSILON and np.linalg.norm(self._e_vec[:,0][3:]) < self.ROT_EPSILON) \
                or (np.linalg.norm(self._e_vec[:,0][:3] - self._prev_e_vec[:3]) < self.POS_EPSILON and np.linalg.norm(self._e_vec[:,0][3:] - self._prev_e_vec[3:]) < self.ROT_EPSILON):
                break
            self._E_mat = 1/2*(self._e_vec.T @ self._W_E_mat @ self._e_vec)
            self._W_N_mat = self._E_mat*np.identity(len(self.DAMP_PARAM)) + self._W_N_bar_mat
            self._J_mat = jacob_pvrm(self.theta, fk_pvrm)
            self._H_mat = self._J_mat.T @ self._W_E_mat @ self._J_mat + self._W_N_mat
            self._g_vec = self._J_mat.T @ self._W_E_mat @ self._e_vec
            self.theta += (np.linalg.inv(self._H_mat) @ self._g_vec)[:,0]
            if is_jrange_mode:
                self.theta = self.theta*(self.theta >= self.JOINT_LIMIT_MIN) + self.JOINT_LIMIT_MIN*(self.theta < self.JOINT_LIMIT_MIN)
                self.theta = self.theta*(self.theta <= self.JOINT_LIMIT_MAX) + self.JOINT_LIMIT_MAX*(self.theta > self.JOINT_LIMIT_MAX)
            self._prev_e_vec = self._e_vec[:,0]
        return self.theta

    # Simple numerical IK for target position vectors from initial theta 
    def ik_pvs_simple(self, init_theta, tgt_pvs, fk_pvs, jacob_pvs, is_jrange_mode):
        self._W_E_mat = np.diag(np.ones(len(tgt_pvs)))
        self._W_N_bar_mat = np.diag(self.DAMP_PARAM)
        self.theta = init_theta
        self._prev_e_vec = np.zeros(len(tgt_pvs))
        while(True):
            self._e_vec = self.error_pvs(self.theta, tgt_pvs, fk_pvs).reshape(-1,1)
            self._e_count = 0
            for i in range(int(len(tgt_pvs)/3)):
                if ((np.linalg.norm(self._e_vec[:,0][3*i:3*(i+1)]) < self.POS_EPSILON) or \
                    (np.linalg.norm(self._e_vec[:,0][3*i:3*(i+1)] - self._prev_e_vec[3*i:3*(i+1)]) < self.POS_EPSILON)):
                    self._e_count += 1
            if self._e_count >= 3:
                break
            self._E_mat = 1/2*(self._e_vec.T @ self._W_E_mat @ self._e_vec)
            self._W_N_mat = self._E_mat*np.identity(len(self.DAMP_PARAM)) + self._W_N_bar_mat
            self._J_mat = jacob_pvs(self.theta, fk_pvs)
            self._H_mat = self._J_mat.T @ self._W_E_mat @ self._J_mat + self._W_N_mat
            self._g_vec = self._J_mat.T @ self._W_E_mat @ self._e_vec
            self.theta += (np.linalg.inv(self._H_mat) @ self._g_vec)[:,0]
            if is_jrange_mode:
                self.theta = self.theta*(self.theta >= self.JOINT_LIMIT_MIN) + self.JOINT_LIMIT_MIN*(self.theta < self.JOINT_LIMIT_MIN)
                self.theta = self.theta*(self.theta <= self.JOINT_LIMIT_MAX) + self.JOINT_LIMIT_MAX*(self.theta > self.JOINT_LIMIT_MAX)
            self._prev_e_vec = self._e_vec[:,0]
        return self.theta

    # IK for single arm 
    def ik_arm_simple(self, init_theta, tgt_pvrm, is_jrange_mode=False):
        self.theta = self.ik_pvrm_simple(init_theta, tgt_pvrm, self.fk.fk_arm_right_pvrm, self.jacob.jacob_pvrm_mat, is_jrange_mode)
        return self.theta

    # IK for three finger positions
    def ik_fingers_simple(self, init_theta, tgt_pvs, is_jrange_mode=False):
        self.theta = self.ik_pvs_simple(init_theta, tgt_pvs, self.fk.fk_fingers_right_pvs, self.jacob.jacob_pvs_mat, is_jrange_mode)
        return self.theta