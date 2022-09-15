#########################################
# Program to evaluate IK (Inverse Kinematics)
# This program needs completed FK (Forward Kinematics)
#########################################

import numpy as np
from scipy.spatial.transform import Rotation as R
import  time

POS_EPSILON = 1e-4
ROT_EPSILON = 1e-6

ARM_LINK_LEN = np.array([0.315, 0., 0.33, 0.02, 0.428, 0.0045, 0.03])
THUMB_LINK_LEN = np.array([78e-3, 15e-3, 6.7e-3, 36e-3, 4.5e-3, 27.5e-3, 40e-3, 20e-3, 20e-3])
INDEX_LINK_LEN = np.array([156.5e-3, 17.5e-3, 6.2e-3, 20e-3, 3e-3, 45.5e-3, 20e-3, 20e-3])
MIDDLE_LINK_LEN = np.array([161.5e-3, 17.5e-3, 18.3e-3, 20e-3, 3e-3, 50.5e-3, 20e-3, 20e-3])

# Calc rotation matrix
def rot_mat(alpha,theta):
    mat = np.array([[np.cos(theta),              -np.sin(theta),               0             ],
                    [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha)],
                    [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha)]])
    return mat

# Calc transform matrix
def trans_mat(a,d,alpha,theta):
    _vec = np.array([[a],[-d*np.sin(alpha)],[d*np.cos(alpha)]])
    mat = np.append(rot_mat(alpha, theta), _vec, axis=1)
    return np.append(mat, np.array([[0,0,0,1]]), axis=0)

# Forward kinematics
def fk(theta):
    # Arm
    _tmA1 = trans_mat(0., ARM_LINK_LEN[0], 0., theta[0])
    _tmA2 = trans_mat(0., ARM_LINK_LEN[1], -np.pi/2, theta[1])
    _tmA3 = trans_mat(0., ARM_LINK_LEN[2], np.pi/2, theta[2])
    _tmA4 = trans_mat(ARM_LINK_LEN[3], 0., -np.pi/2, theta[3])
    _tmA5 = trans_mat(0., ARM_LINK_LEN[4], np.pi/2, theta[4] + np.pi/2)
    _tmA6 = trans_mat(ARM_LINK_LEN[5], 0., np.pi/2, theta[5] + np.pi/2)
    _tmA7 = trans_mat(ARM_LINK_LEN[6], 0., np.pi/2, theta[6])
    _tmAE = trans_mat(0., 0, -np.pi/2, np.pi)
    _tmA = _tmA1 @ _tmA2 @ _tmA3 @ _tmA4 @ _tmA5 @ _tmA6 @ _tmA7 @_tmAE
    # Thumb
    _tmT0 = trans_mat(-THUMB_LINK_LEN[0], THUMB_LINK_LEN[1], 0., np.pi/2)
    _tmT1 = trans_mat(THUMB_LINK_LEN[2], 0., np.pi/2, theta[7])
    _tmT2 = trans_mat(0., THUMB_LINK_LEN[3], -np.pi/2, theta[8] + np.pi/2)
    _tmT3 = trans_mat(THUMB_LINK_LEN[4], 0., -np.pi/2, theta[9])
    _tmT3p = trans_mat(THUMB_LINK_LEN[5], 0., 0., -np.pi/2)
    _tmT4 = trans_mat(THUMB_LINK_LEN[6], 0., 0., theta[10] + np.pi/2)
    _tmTE = trans_mat(THUMB_LINK_LEN[7], THUMB_LINK_LEN[8], np.pi/2, 0.)
    _tmT = _tmA @ _tmT0 @ _tmT1 @ _tmT2 @ _tmT3 @ _tmT3p @ _tmT4 @ _tmTE
    posT = _tmT[:3,3]
    rotT = R.from_matrix(_tmT[:3,:3]).as_rotvec()
    # Index
    _tmI0 = trans_mat(-INDEX_LINK_LEN[0], INDEX_LINK_LEN[1], 0., np.pi/2)
    _tmI1 = trans_mat(INDEX_LINK_LEN[2], 0., np.pi, theta[11] + np.pi/2)
    _tmI2 = trans_mat(-INDEX_LINK_LEN[3], 0., np.pi/2, theta[12] - np.pi/2)
    _tmI2p = trans_mat(INDEX_LINK_LEN[4], 0., 0., -np.pi/2)
    _tmI3 = trans_mat(INDEX_LINK_LEN[5], 0., 0., theta[13] + np.pi/2)
    _tmIE = trans_mat(INDEX_LINK_LEN[6], INDEX_LINK_LEN[7], np.pi/2, 0.)
    _tmI = _tmA @ _tmI0 @ _tmI1 @ _tmI2 @ _tmI2p @ _tmI3 @ _tmIE
    posI = _tmI[:3,3]
    rotI = R.from_matrix(_tmI[:3,:3]).as_rotvec()
    # Index
    _tmM0 = trans_mat(-MIDDLE_LINK_LEN[0], MIDDLE_LINK_LEN[1], 0., np.pi/2)
    _tmM1 = trans_mat(-MIDDLE_LINK_LEN[2], 0., np.pi, theta[14] + np.pi/2)
    _tmM2 = trans_mat(-MIDDLE_LINK_LEN[3], 0., np.pi/2, theta[15] - np.pi/2)
    _tmM2p = trans_mat(MIDDLE_LINK_LEN[4], 0., 0., -np.pi/2)
    _tmM3 = trans_mat(MIDDLE_LINK_LEN[5], 0., 0., theta[16] + np.pi/2)
    _tmME = trans_mat(MIDDLE_LINK_LEN[6], MIDDLE_LINK_LEN[7], np.pi/2, 0.)
    _tmM = _tmA @ _tmM0 @ _tmM1 @ _tmM2 @ _tmM2p @ _tmM3 @ _tmME
    posM = _tmM[:3,3]
    rotM = R.from_matrix(_tmM[:3,:3]).as_rotvec()
    return np.concatenate([posT, posI, posM, rotT, rotI, rotM])

# Jacobian matrix
def jacob_mat(theta, output_len):
    _epsilon = 1e-6
    _l_theta = len(theta)
    J_mat = np.zeros((output_len,_l_theta))
    for i in range(_l_theta):
        _theta_p = theta + _epsilon/2*np.identity(_l_theta)[i,:]
        _theta_m = theta - _epsilon/2*np.identity(_l_theta)[i,:]
        J_mat[:,i] = (fk(_theta_p)[:output_len] - fk(_theta_m)[:output_len])/_epsilon
    return J_mat

# Inverse kinematics
def ik(tgt_pose, output_len):
    POS_EPSILON = 1e-4
    ROT_EPSILON = 1e-6
    _theta_init = np.zeros(17)
    _W_E_mat = np.diag(np.ones(output_len))
    _ARM_LEN = np.sum(ARM_LINK_LEN)
    _THUMB_LEN = np.sum(THUMB_LINK_LEN)
    _INDEX_LEN = np.sum(INDEX_LINK_LEN)
    _MIDDLE_LEN = np.sum(MIDDLE_LINK_LEN)
    _damp_param = np.concatenate([_ARM_LEN*1e-3*np.ones(7), _ARM_LEN*1e-3*np.ones(4), 
                                _INDEX_LEN*1e-3*np.ones(3), _MIDDLE_LEN*1e-3*np.ones(3)])
    _W_N_bar_mat = np.diag(_damp_param)
    theta = _theta_init
    while(True):
        _e_vec = (tgt_pose - fk(theta)[:output_len]).reshape(-1,1)
        _E_mat = 1/2*(np.transpose(_e_vec) @ _W_E_mat @ _e_vec)
        _W_N_mat = _E_mat*np.identity(len(_damp_param)) + _W_N_bar_mat
        _J_mat = jacob_mat(theta, output_len)
        _H_mat = np.transpose(_J_mat) @ _W_E_mat @ _J_mat + _W_N_mat
        _g_vec = np.transpose(_J_mat) @ _W_E_mat @ _e_vec
        theta += (np.linalg.inv(_H_mat) @ _g_vec)[:,0]
        _error = _e_vec[:,0]
        if (np.linalg.norm(_error[:3]) < POS_EPSILON) and (np.linalg.norm(_error[3:6]) < POS_EPSILON) \
                and (np.linalg.norm(_error[6:]) < POS_EPSILON):
            break
    return theta

if __name__ == '__main__':
    output_len = 9
    for j in range(100):
        theta_act = np.random.uniform(low=-np.pi, high=np.pi, size=17)
        tgt_pose = fk(theta_act)[:output_len]
        start_time = time.perf_counter()
        theta_est = ik(tgt_pose, output_len)
        end_time = time.perf_counter()
        ik_pose = fk(theta_est)[:output_len]
        result = ['OK', 0.]
        for i in range(3):
            pose_error = np.linalg.norm(tgt_pose[3*i:3*(i+1)] - ik_pose[3*i:3*(i+1)])
            if pose_error >= POS_EPSILON:
                result[0] = 'NG'
            if pose_error > result[1]:
                result[1] = pose_error
        print('=========================================')
        print('Test No.', j+1)
        print('Result:', result[0])
        print('Max error [mm]:', result[1]*1e3)
        print('Performance time [ms]:', (end_time - start_time)*1e3)