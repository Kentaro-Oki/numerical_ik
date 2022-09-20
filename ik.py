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
        e_pv = tgt_pvrm[0] - fk_pvrm(theta)[0]
        e_rv = self.jacob.error_rms(fk_pvrm(theta)[1], tgt_pvrm[1])
        return np.concatenate((e_pv, e_rv))

    # Calc. error of position vector and rotation matrix from theta of current pose and target position vector and rotation matrix
    def error_pvs(self, theta, tgt_pvs, fk_pvs):
        e_pvs = tgt_pvs[:3] - fk_pvs(theta)[:3]
        for i in range(int(len(tgt_pvs)/3)-1):
            _tmp_err = tgt_pvs[3*(i+1):3*(i+2)] - fk_pvs(theta)[3*(i+1):3*(i+2)]
            e_pvs = np.append(e_pvs, _tmp_err)
        return e_pvs

    # Convert a rotation vector to an unit quaternion for calculate lambda in prioritized IK
    def error_pvrm_update_quat_lambda(self, theta, tgt_pvrm, fk_pvrm, lamb):
        _e_vec = self.error_pvrm(theta, tgt_pvrm, fk_pvrm)
        _e_pv = _e_vec[:3]
        _e_rv = _e_vec[3:]
        _e_quat = self.fk.rv2uq(_e_rv)
        next_lamb = np.zeros(len(lamb))
        next_lamb[:3] = lamb[:3] + _e_pv
        next_lamb[3:] = (lamb[3:] + _e_quat)/np.linalg.norm(lamb[3:] + _e_quat)
        e_vec = np.concatenate((_e_pv, R.from_quat(lamb[3:]).as_rotvec()), axis=0)
        return _e_vec, next_lamb

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
            self._J_mat = jacob_pvrm(self.theta)
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
            self._e_no_update_count = 0
            for i in range(int(len(tgt_pvs)/3)):
                if ((np.linalg.norm(self._e_vec[:,0][3*i:3*(i+1)]) < self.POS_EPSILON) or \
                    (np.linalg.norm(self._e_vec[:,0][3*i:3*(i+1)] - self._prev_e_vec[3*i:3*(i+1)]) < self.POS_EPSILON)):
                    self._e_no_update_count += 1
            if self._e_no_update_count >= int(len(tgt_pvs)):
                break
            self._E_mat = 1/2*(self._e_vec.T @ self._W_E_mat @ self._e_vec)
            self._W_N_mat = self._E_mat*np.identity(len(self.DAMP_PARAM)) + self._W_N_bar_mat
            self._J_mat = jacob_pvs(self.theta)
            self._H_mat = self._J_mat.T @ self._W_E_mat @ self._J_mat + self._W_N_mat
            self._g_vec = self._J_mat.T @ self._W_E_mat @ self._e_vec
            self.theta += (np.linalg.inv(self._H_mat) @ self._g_vec)[:,0]
            if is_jrange_mode:
                self.theta = self.theta*(self.theta >= self.JOINT_LIMIT_MIN) + self.JOINT_LIMIT_MIN*(self.theta < self.JOINT_LIMIT_MIN)
                self.theta = self.theta*(self.theta <= self.JOINT_LIMIT_MAX) + self.JOINT_LIMIT_MAX*(self.theta > self.JOINT_LIMIT_MAX)
            self._prev_e_vec = self._e_vec[:,0]
        return self.theta

    # Prioritized IK for a target set of position vector and rotation matrix from initial theta
    def ik_pvrm_pvs_prioritized(self, init_theta, tgt_pvrm, sub_tgt_pvs, fk_pvrm, sub_fk_pvs, jacob_pvrm, sub_jacob_pvs, is_jrange_mode):
        self._W_W_mat = np.diag(np.ones(len(sub_tgt_pvs)))
        self._W_E_mat = np.concatenate((np.concatenate((np.identity(6), np.zeros((6,len(sub_tgt_pvs)))), axis=1), np.concatenate((np.zeros((len(sub_tgt_pvs),6)), self._W_W_mat), axis=1)), axis=0)
        self._W_N_bar_mat = np.diag(self.DAMP_PARAM)
        self.theta = init_theta
        self._lambda = np.array([0,0,0,0,0,0,1]) # zero positional error + unit quaternion of zero rotation
        self._prev_e_p_vec = np.zeros(int(len(tgt_pvrm) + len(sub_tgt_pvs)))
        while(True):
            self._e_S_vec, self._lambda = self.error_pvrm_update_quat_lambda(self.theta, tgt_pvrm, fk_pvrm, self._lambda)
            self._e_S_vec = self._e_S_vec.reshape(-1,1)
            self._e_W_vec = self.error_pvs(self.theta, sub_tgt_pvs, sub_fk_pvs).reshape(-1,1)
            self._e_W_no_update_count = 0
            for i in range(int(len(sub_tgt_pvs)/3)):
                if ((np.linalg.norm(self._e_W_vec[:,0][3*(i+2):3*(i+3)]) < self.POS_EPSILON) or \
                    (np.linalg.norm(self._e_W_vec[:,0][3*(i+2):3*(i+3)] - self._prev_e_p_vec[3*(i+2):3*(i+3)]) < self.POS_EPSILON)):
                    self._e_W_no_update_count += 1
            if (np.linalg.norm(self._e_S_vec[:,0][:3]) < self.POS_EPSILON and np.linalg.norm(self._e_S_vec[:,0][3:]) < self.ROT_EPSILON) \
                or ((np.linalg.norm(self._e_S_vec[:,0][:3] - self._prev_e_p_vec[:3]) < self.POS_EPSILON and np.linalg.norm(self._e_S_vec[:,0][3:] - self._prev_e_p_vec[3:6]) < self.ROT_EPSILON) \
                and self._e_W_no_update_count >= int(len(sub_tgt_pvs)/3)):
                break
            self._e_p_vec = np.concatenate((self._e_S_vec, self._e_W_vec), axis=0)
            self._E_mat = 1/2*self._e_p_vec.T @ self._W_E_mat @ self._e_p_vec
            self._W_N_mat = self._E_mat*np.identity(len(self.DAMP_PARAM)) + self._W_N_bar_mat
            self._J_S_mat = jacob_pvrm(self.theta)
            self._J_W_mat = sub_jacob_pvs(self.theta)
            self._J_mat = np.concatenate((self._J_S_mat, self._J_W_mat), axis=0)
            self._H_mat = self._J_mat.T @ self._W_E_mat @ self._J_mat + self._W_N_mat
            self._g_vec = self._J_mat.T @ self._W_E_mat @ self._e_p_vec
            self.theta += (np.linalg.inv(self._H_mat) @ self._g_vec)[:,0]
            if is_jrange_mode:
                self.theta = self.theta*(self.theta >= self.JOINT_LIMIT_MIN) + self.JOINT_LIMIT_MIN*(self.theta < self.JOINT_LIMIT_MIN)
                self.theta = self.theta*(self.theta <= self.JOINT_LIMIT_MAX) + self.JOINT_LIMIT_MAX*(self.theta > self.JOINT_LIMIT_MAX)
            self._prev_e_p_vec = self._e_p_vec[:,0]
        return self.theta

    # Prioritized IK for a target set of position vectors from initial theta
    def ik_pvs_pvs_prioritized(self, init_theta, tgt_pvs, sub_tgt_pvs, fk_pvs, sub_fk_pvs, jacob_pvs, sub_jacob_pvs, is_jrange_mode):
        self._W_W_mat = np.diag(np.ones(len(sub_tgt_pvs)))
        self._W_E_mat = np.concatenate((np.concatenate((np.identity(len(tgt_pvs)), np.zeros((len(tgt_pvs),len(sub_tgt_pvs)))), axis=1), np.concatenate((np.zeros((len(sub_tgt_pvs),len(tgt_pvs))), self._W_W_mat), axis=1)), axis=0)
        self._W_N_bar_mat = np.diag(self.DAMP_PARAM)
        self.theta = init_theta
        self._lambda = np.zeros(len(tgt_pvs)) # unit quaternion of zero rotation
        self._prev_e_p_vec = np.zeros(len(tgt_pvs) + len(sub_tgt_pvs))
        while(True):
            self._e_S_vec = self.error_pvs(self.theta, tgt_pvs, fk_pvs).reshape(-1,1)
            self._e_W_vec = self.error_pvs(self.theta, sub_tgt_pvs, sub_fk_pvs).reshape(-1,1)
            self._e_p_vec = np.concatenate((self._e_S_vec, self._e_W_vec), axis=0)
            self._e_W_no_update_count = 0
            for i in range(int((len(tgt_pvs) + len(sub_tgt_pvs))/3)):
                if ((np.linalg.norm(self._e_p_vec[:,0][3*i:3*(i+1)]) < self.POS_EPSILON) or \
                    (np.linalg.norm(self._e_p_vec[:,0][3*i:3*(i+1)] - self._prev_e_p_vec[3*i:3*(i+1)]) < self.POS_EPSILON)):
                    self._e_W_no_update_count += 1
            if (self._e_W_no_update_count >= int((len(tgt_pvs) + len(sub_tgt_pvs))/3)):
                break
            self._E_mat = 1/2*self._e_p_vec.T @ self._W_E_mat @ self._e_p_vec
            self._W_N_mat = self._E_mat*np.identity(len(self.DAMP_PARAM)) + self._W_N_bar_mat
            self._J_S_mat = jacob_pvs(self.theta)
            self._J_W_mat = sub_jacob_pvs(self.theta)
            self._J_mat = np.concatenate((self._J_S_mat, self._J_W_mat), axis=0)
            self._H_mat = self._J_mat.T @ self._W_E_mat @ self._J_mat + self._W_N_mat
            self._g_vec = self._J_mat.T @ self._W_E_mat @ self._e_p_vec
            self.theta += (np.linalg.inv(self._H_mat) @ self._g_vec)[:,0]
            if is_jrange_mode:
                self.theta = self.theta*(self.theta >= self.JOINT_LIMIT_MIN) + self.JOINT_LIMIT_MIN*(self.theta < self.JOINT_LIMIT_MIN)
                self.theta = self.theta*(self.theta <= self.JOINT_LIMIT_MAX) + self.JOINT_LIMIT_MAX*(self.theta > self.JOINT_LIMIT_MAX)
            self._prev_e_p_vec = self._e_p_vec[:,0]
        return self.theta


    # IK for single arm 
    def ik_arm_simple(self, init_theta, tgt_pvrm, is_jrange_mode=False):
        self.theta = self.ik_pvrm_simple(init_theta, tgt_pvrm, self.fk.fk_arm_right_pvrm, self.jacob.arm, is_jrange_mode)
        return self.theta

    # IK for three finger positions
    def ik_fingers_simple(self, init_theta, tgt_pvs, is_jrange_mode=False):
        self.theta = self.ik_pvs_simple(init_theta, tgt_pvs, self.fk.fk_fingers_right_pvs, self.jacob.finger_tips, is_jrange_mode)
        return self.theta

    # IK for single arm with prioritized
    def ik_arm_prioritized(self, init_theta, tgt_pvrm, sub_tgt_pvs, is_jrange_mode=False):
        self.theta = self.ik_pvrm_pvs_prioritized(init_theta, tgt_pvrm, sub_tgt_pvs, self.fk.fk_arm_right_pvrm, self.fk.fk_elbow_wrist_right_pvs, self.jacob.arm, self.jacob.elbow_wrist, is_jrange_mode)
        return self.theta

    # IK for three finger positions with prioritized
    def ik_fingers_prioritized(self, init_theta, tgt_pvs, sub_tgt_pvs, is_jrange_mode=False):
        self.theta = self.ik_pvs_pvs_prioritized(init_theta, tgt_pvs, sub_tgt_pvs, self.fk.fk_fingers_right_pvs, self.fk.fk_elbow_wrist_right_pvs, self.jacob.finger_tips, self.jacob.elbow_wrist, is_jrange_mode)
        return self.theta