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

def fk_elbow_pos(theta):
    _tm1 = trans_mat(0., LINK_LEN[0], 0., theta[0])
    _tm2 = trans_mat(0., LINK_LEN[1], -np.pi/2, theta[1])
    _tm3 = trans_mat(0., LINK_LEN[2], np.pi/2, theta[2])
    _tm4 = trans_mat(LINK_LEN[3], 0., -np.pi/2, theta[3])
    _tm = _tm1 @ _tm2 @ _tm3 @ _tm4
    pos = _tm[:3,3]
    return pos

def fk_wrist_pos(theta):
    _tm1 = trans_mat(0., LINK_LEN[0], 0., theta[0])
    _tm2 = trans_mat(0., LINK_LEN[1], -np.pi/2, theta[1])
    _tm3 = trans_mat(0., LINK_LEN[2], np.pi/2, theta[2])
    _tm4 = trans_mat(LINK_LEN[3], 0., -np.pi/2, theta[3])
    _tm5 = trans_mat(0., LINK_LEN[4], np.pi/2, theta[4] + np.pi/2)
    _tm6 = trans_mat(LINK_LEN[5], 0., np.pi/2, theta[5] + np.pi/2)
    _tm = _tm1 @ _tm2 @ _tm3 @ _tm4 @ _tm5 @ _tm6
    pos = _tm[:3,3]
    return pos

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

def jacob_elbow_mat(theta):
    _epsilon = 1e-8
    _l_theta = len(theta)
    J_mat = np.zeros((3,_l_theta))
    for i in range(_l_theta):
        _theta_p = theta + _epsilon/2*np.identity(_l_theta)[i,:]
        _theta_m = theta - _epsilon/2*np.identity(_l_theta)[i,:]
        J_mat[:,i] = (fk_elbow_pos(_theta_p) - fk_elbow_pos(_theta_m))/_epsilon
    return J_mat

def jacob_wrist_mat(theta):
    _epsilon = 1e-8
    _l_theta = len(theta)
    J_mat = np.zeros((3,_l_theta))
    for i in range(_l_theta):
        _theta_p = theta + _epsilon/2*np.identity(_l_theta)[i,:]
        _theta_m = theta - _epsilon/2*np.identity(_l_theta)[i,:]
        J_mat[:,i] = (fk_wrist_pos(_theta_p) - fk_wrist_pos(_theta_m))/_epsilon
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
    _LINK_LEN_SUM = np.sum(LINK_LEN)
    _W_E_mat = np.diag(np.array([1/_LINK_LEN_SUM, 1/_LINK_LEN_SUM, 1/_LINK_LEN_SUM, 1/(2*np.pi), 1/(2*np.pi), 1/(2*np.pi)]))
    _damp_param = (_LINK_LEN_SUM**2)*1e-3*np.ones(7)
    _W_N_bar_mat = np.diag(_damp_param)
    theta = _theta_init
    for _ in range(20):
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

# Inverse kinematics with joint range constraints
def ik_collision(tgt_pose, tgt_elbow_pos, tgt_wrist_pos, theta_init):
    # _W_E_mat = np.diag(np.concatenate((np.ones(len(tgt_pose)), _weight_param)))
    _LINK_LEN_SUM = np.sum(LINK_LEN)
    _W_E_mat = np.diag(np.array([1/_LINK_LEN_SUM, 1/_LINK_LEN_SUM, 1/_LINK_LEN_SUM, 1/(2*np.pi), 1/(2*np.pi), 1/(2*np.pi)]))
    _LINK_LEN_SUM = np.sum(LINK_LEN)
    _damp_param = (_LINK_LEN_SUM**2)*1e-3*np.ones(len(theta_init))
    _W_N_bar_mat = np.diag(_damp_param)
    _weight_param = np.array([1,1,1,1,1,1])
    _W_w_mat = np.diag(_weight_param)
    _lambda = np.zeros(len(tgt_pose)).reshape(-1,1)
    theta = theta_init
    for _ in range(100):
        _e_s_vec = (tgt_pose - fk(theta)).reshape(-1,1)
        _E_mat = 1/2*(np.transpose(_e_s_vec) @ _W_E_mat @ _e_s_vec)
        _lambda += _e_s_vec
        _e_elbow_vec = tgt_elbow_pos - fk_elbow_pos(theta)
        _e_wrist_vec = tgt_wrist_pos - fk_wrist_pos(theta)
        _e_w_vec = np.concatenate((_e_elbow_vec, _e_wrist_vec)).reshape(-1,1)
        _e_p_vec = np.concatenate((_e_s_vec + _lambda, _e_w_vec), axis=0)
        _W_N_mat = _E_mat*np.identity(len(_damp_param)) + _W_N_bar_mat
        _J_s_mat = jacob_mat(theta)
        _J_elbow_mat = jacob_elbow_mat(theta)
        _J_wrist_mat = jacob_wrist_mat(theta)
        _J_w_mat = np.concatenate((_J_elbow_mat, _J_wrist_mat), axis=0)
        _J_mat = np.concatenate((_J_s_mat, _J_w_mat), axis=0)
        _H_mat = _J_s_mat.T @ _J_s_mat + _J_w_mat.T @ _W_w_mat @ _J_w_mat + _W_N_mat
        _g_vec = _J_s_mat.T @ _lambda + _J_w_mat.T @ _W_w_mat @ _e_w_vec
        theta += (np.linalg.inv(_H_mat) @ _g_vec)[:,0]
        # theta = theta*(theta > J_RANGE_MIN) + _theta_init*(theta<=J_RANGE_MIN)
        # theta = theta*(theta < J_RANGE_MAX) + _theta_init*(theta>=J_RANGE_MAX)
        # _error = tgt_pose - fk(theta)
        # if (np.linalg.norm(_error[:3]) < POS_EPSILON) and (np.linalg.norm(_error[3:]) < ROT_EPSILON):
        #     break
    return theta

if __name__ == '__main__':
    score = 0
    for i in range(1):
        # theta_act = np.random.uniform(low=J_RANGE_MIN, high=J_RANGE_MAX)
        theta_act = (J_RANGE_MIN + J_RANGE_MAX)/2 + 1e-2*np.ones(7)
        tgt_pose = fk(theta_act)
        tgt_elbow_pos = np.array([0.5, 0., 1.0])
        tgt_wrist_pos = np.array([0.5, 0., 1.0])
        theta_est = ik(tgt_pose)
        ik_pose = fk(theta_est)
        theta_est_range = ik_range(tgt_pose)
        ik_pose_range = fk(theta_est_range)
        theta_est_collision = ik_collision(tgt_pose, tgt_elbow_pos, tgt_wrist_pos, theta_est_range)
        ik_pose_collision = fk(theta_est_collision)
        error_range = tgt_pose - ik_pose_range
        error_collision = tgt_pose - ik_pose_collision
        print('===========================')
        print('Test No.', i+1)
        print('Actual theta:', np.degrees(theta_act))
        print('Target pose:', tgt_pose)
        print('Estimated theta:', theta_est)
        print('IK pose:', ik_pose)
        print('Estimated theta with collision:', np.degrees(theta_est_collision))
        print('IK pose with joint collision:', ik_pose_collision)
        print('Range Error (Positon, Rotation):', np.linalg.norm(error_range[:3]), np.linalg.norm(error_range[3:]))
        if (np.linalg.norm(error_range[:3]) < POS_EPSILON) and (np.linalg.norm(error_range[3:]) < ROT_EPSILON):
            print('Range:', colored('OK', 'green'))
            score += 1
        else:
            print('Range:', colored('NG', 'red'))
        print('Collision Error (Positon, Rotation):', np.linalg.norm(error_collision[:3]), np.linalg.norm(error_collision[3:]))
        if (np.linalg.norm(error_collision[:3]) < POS_EPSILON) and (np.linalg.norm(error_collision[3:]) < ROT_EPSILON):
            print('Collision:', colored('OK', 'green'))
            score += 1
        else:
            print('Collision:', colored('NG', 'red'))
    print('===========================')
    print('Score:', score)