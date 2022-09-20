import numpy as np 
from fk import FK
from ik import IK

if __name__ == '__main__':
    fk = FK()
    ik = IK()
    init_theta = (ik.JOINT_LIMIT_MIN + ik.JOINT_LIMIT_MAX)/2
    tgt_theta = init_theta + np.pi/6*np.ones(17)
    tgt_pvs = fk.fk_fingers_right_pvs(tgt_theta)
    ik_theta = ik.ik_fingers_simple(init_theta, tgt_pvs, True)
    tgt_pvs = fk.fk_fingers_right_pvs(tgt_theta)
    ik_pvs = fk.fk_fingers_right_pvs(ik_theta)
    print('Target theta:', tgt_theta)
    print('IK theta:', ik_theta)
    print('Target PVS:', tgt_pvs)
    print('IK PVS:', ik_pvs)