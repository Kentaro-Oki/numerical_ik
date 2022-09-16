import numpy as np
from scipy.spatial.transform import Rotation as R
from dh_param import DH_param

class FK:
    def __init__(self):
        self.dh = DH_param()

    def rot_mat(self, alpha, theta):
        return np.array([[np.cos(theta),              -np.sin(theta),               0             ],
                        [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha)],
                        [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha)]])

    def trans_mat(self, dh):
        self._pv = np.array([[dh[0]],[-dh[1]*np.sin(dh[2])],[dh[1]*np.cos(dh[2])]]) # position vector
        self._rm = np.append(self.rot_mat(dh[2], dh[3]), self._pv, axis=1) # rotation matrix
        return np.append(self._rm, np.array([[0,0,0,1]]), axis=0)

    def tm_chain_part(self, dh, link_no):
        self.tm = np.identity(4)
        for i in range(link_no):
            self.tm = self.tm @ self.trans_mat(dh[i,:])
        return self.tm

    def tm_arm(self, theta):
        self.dh_arm = self.dh.arm_right(theta)
        return self.tm_chain_part(self.dh_arm, self.dh_arm.shape[0])

    def tm_elbow(self, theta):
        return self.tm_chain_part(self.dh.arm_right(theta), 5)

    def tm_wrist(self, theta):
        return self.tm_chain_part(self.dh.arm_right(theta), 7)

    def tm_arm_thumb_right(self, theta):
        self.dh_thumb = self.dh.thumb_right(theta)
        return self.tm_chain_part(self.dh_thumb, self.dh_thumb.shape[0])

    def tm_arm_index_right(self, theta):
        self.dh_index = self.dh.index_right(theta)
        return self.tm_chain_part(self.dh_index, self.dh_index.shape[0])

    def tm_arm_middle_right(self, theta):
        self.dh_middle = self.dh.middle_right(theta)
        return self.tm_chain_part(self.dh_middle, self.dh_middle.shape[0])

    # Transform matrix to position vector
    def tm2pv(self, tm):
        return tm[:3,3]

    # Transform matrix to position vector and rotation matrix
    def tm2pvrm(self, tm):
        return tm[:3,3], tm[:3,:3]

    # Rotation matrix to Rotation vector
    def rm2rv(self, rm):
        return R.from_matrix(rm).as_rotvec()

    # Rotation vector to Rotation matrix
    def rv2rm(self, rv):
        return R.from_rotvec(rv).as_matrix()

    # FK of world to right arm tip with output as position vector and rotation matrix
    def fk_arm_right_pvrm(self, theta):
        return self.tm2pvrm(self.tm_arm(theta))

    # FK of world to right elbow with output as position vector and rotation matrix
    def fk_elbow_right_pvrm(self, theta):
        return self.tm2pvrm(self.tm_elbow(theta))

    # FK of world to right wrist with output as position vector and rotation matrix
    def fk_wrist_right_pvrm(self, theta):
        return self.tm2pvrm(self.tm_wrist(theta))

    # FK of world to right arm tip with output as position vector and rotation vector
    def fk_arm_right_pvrv(self, theta):
        self.pv, self._rm = self.fk_arm_right_pvrm(theta)
        return self.pv, self.rm2rv(self._rm)

    # FK of world to right thumb with output as position vector and rotation matrix
    def fk_thumb_right_pvrm(self, theta):
        return self.tm2pvrm(self.tm_arm(theta) @ self.tm_arm_thumb_right(theta))

    # FK of world to right thumb with output as position vector and rotation matrix
    def fk_index_right_pvrm(self, theta):
        return self.tm2pvrm(self.tm_arm(theta) @ self.tm_arm_index_right(theta))

    # FK of world to right thumb with output as position vector and rotation matrix
    def fk_middle_right_pvrm(self, theta):
        return self.tm2pvrm(self.tm_arm(theta) @ self.tm_arm_middle_right(theta))

    # FK of world to right finger tips with output as position vectors
    def fk_fingers_right_pvs(self, theta):
        self._tm_arm = self.tm_arm(theta)
        self.pv_thumb = self.tm2pv(self._tm_arm @ self.tm_arm_thumb_right(theta))
        self.pv_index = self.tm2pv(self._tm_arm @ self.tm_arm_index_right(theta))
        self.pv_middle = self.tm2pv(self._tm_arm @ self.tm_arm_middle_right(theta))
        return self.pv_thumb, self.pv_index, self.pv_middle