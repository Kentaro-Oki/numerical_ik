import numpy as np
from scipy.spatial.transform import Rotation as R
import  time
from termcolor import colored

POS_EPSILON = 1e-4
ROT_EPSILON = np.radians(0.1)
LINK_LEN = np.array([0.315, 0., 0.33, 0.02, 0.428, 0.0045, 0.03])
J_RANGE_MIN = np.radians([-190., -90., -360., -15., -360., -60., -70.])
J_RANGE_MAX = np.radians([190., 90., 360., 180., 360., 60., 70.])

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
    _tm1 = trans_mat(0., LINK_LEN[0], 0., theta[0])
    _tm2 = trans_mat(0., LINK_LEN[1], -np.pi/2, theta[1])
    _tm3 = trans_mat(0., LINK_LEN[2], np.pi/2, theta[2])
    _tm4 = trans_mat(LINK_LEN[3], 0., -np.pi/2, theta[3])
    _tm5 = trans_mat(0., LINK_LEN[4], np.pi/2, theta[4] + np.pi/2)
    _tm6 = trans_mat(LINK_LEN[5], 0., np.pi/2, theta[5] + np.pi/2)
    _tm7 = trans_mat(LINK_LEN[6], 0., np.pi/2, theta[6])
    _tmE = trans_mat(0., 0, -np.pi/2, np.pi)
    _tm = _tm1 @ _tm2 @ _tm3 @ _tm4 @ _tm5 @ _tm6 @ _tm7 @_tmE
    pos = _tm[:3,3]
    rot = R.from_matrix(_tm[:3,:3]).as_rotvec()
    return np.append(pos, rot)

# Jacobian matrix
def jacob_mat(theta):
    _epsilon = 1e-8
    _l_theta = len(theta)
    J_mat = np.zeros((6,_l_theta))
    for i in range(_l_theta):
        _theta_p = theta + _epsilon/2*np.identity(_l_theta)[i,:]
        _theta_m = theta - _epsilon/2*np.identity(_l_theta)[i,:]
        J_mat[:,i] = (fk(_theta_p) - fk(_theta_m))/_epsilon
    return J_mat

# Inverse kinematics with joint range constraints
def ik(tgt_pose):
    _theta_init = (J_RANGE_MAX + J_RANGE_MIN)/2
    _W_E_mat = np.diag(np.array([1, 1, 1, 1, 1, 1]))
    _LINK_LEN_SUM = np.sum(LINK_LEN)
    _damp_param = (_LINK_LEN_SUM**2)*1e-3*np.ones(7)
    _W_N_bar_mat = np.diag(_damp_param)
    theta = _theta_init
    for _ in range(1000):
        _e_vec = (tgt_pose - fk(theta)).reshape(-1,1)
        _E_mat = 1/2*(np.transpose(_e_vec) @ _W_E_mat @ _e_vec)
        _W_N_mat = _E_mat*np.identity(len(_damp_param)) + _W_N_bar_mat
        _J_mat = jacob_mat(theta)
        _H_mat = np.transpose(_J_mat) @ _W_E_mat @ _J_mat + _W_N_mat
        _g_vec = np.transpose(_J_mat) @ _W_E_mat @ _e_vec
        theta += (np.linalg.inv(_H_mat) @ _g_vec)[:,0]
        _error = _e_vec[:,0]
        if (np.linalg.norm(_error[:3]) < POS_EPSILON) and (np.linalg.norm(_error[3:]) < ROT_EPSILON):
            break
    return theta

# Inverse kinematics with joint range constraints
def ik_range(tgt_pose):
    _theta_init = (J_RANGE_MAX + J_RANGE_MIN)/2
    _W_E_mat = np.diag(np.array([1, 1, 1, 1, 1, 1]))
    _LINK_LEN_SUM = np.sum(LINK_LEN)
    _damp_param = (_LINK_LEN_SUM**2)*1e-3*np.ones(7)
    _W_N_bar_mat = np.diag(_damp_param)
    theta = _theta_init
    for _ in range(100):
        _e_vec = (tgt_pose - fk(theta)).reshape(-1,1)
        _E_mat = 1/2*(np.transpose(_e_vec) @ _W_E_mat @ _e_vec)
        _W_N_mat = _E_mat*np.identity(len(_damp_param)) + _W_N_bar_mat
        _J_mat = jacob_mat(theta)
        _H_mat = np.transpose(_J_mat) @ _W_E_mat @ _J_mat + _W_N_mat
        _g_vec = np.transpose(_J_mat) @ _W_E_mat @ _e_vec
        theta += (np.linalg.inv(_H_mat) @ _g_vec)[:,0]
        theta = theta*(theta > J_RANGE_MIN) + _theta_init*(theta<=J_RANGE_MIN)
        theta = theta*(theta < J_RANGE_MAX) + _theta_init*(theta>=J_RANGE_MAX)
        _error = tgt_pose - fk(theta)
        if (np.linalg.norm(_error[:3]) < POS_EPSILON) and (np.linalg.norm(_error[3:]) < ROT_EPSILON):
            break
    return theta

if __name__ == '__main__':
    score = 0
    for i in range(100):
        theta_act = np.random.uniform(low=J_RANGE_MIN, high=J_RANGE_MAX)
        tgt_pose = fk(theta_act)
        theta_est = ik(tgt_pose)
        ik_pose = fk(theta_est)
        theta_est_range = ik_range(tgt_pose)
        ik_pose_range = fk(theta_est_range)
        error = tgt_pose - ik_pose_range
        print('===========================')
        print('Test No.', i+1)
        print('Actual theta:', np.degrees(theta_act))
        print('Target pose:', tgt_pose)
        print('Estimated theta:', theta_est)
        print('IK pose:', ik_pose)
        print('Estimated theta with joint range:', np.degrees(theta_est_range))
        print('IK pose with joint range:', ik_pose_range)
        print('Error (Positon, Rotation):', np.linalg.norm(error[:3]), np.linalg.norm(error[3:]))
        if (np.linalg.norm(error[:3]) < POS_EPSILON) and (np.linalg.norm(error[3:]) < ROT_EPSILON):
            print(colored('OK', 'green'))
            score += 1
        else:
            print(colored('NG', 'red'))
    print('===========================')
    print('Score:', score)