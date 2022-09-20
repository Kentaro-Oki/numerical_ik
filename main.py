import numpy as np 
from fk import FK
from ik import IK

if __name__ == '__main__':
    fk = FK()
    ik = IK()
    init_theta = (ik.JOINT_LIMIT_MIN + ik.JOINT_LIMIT_MAX)/2
    tgt_theta = init_theta - np.pi/3*np.ones(17)

    tgt_pvs = fk.fk_fingers_right_pvs(tgt_theta)
    sub_tgt_pvs = fk.fk_elbow_wrist_right_pvs(tgt_theta)
    sub_tgt_pvs += 0.5*np.ones(len(sub_tgt_pvs))
    ik_theta = ik.ik_fingers_prioritized(init_theta, tgt_pvs, sub_tgt_pvs, True)
    ik_pvs = fk.fk_fingers_right_pvs(ik_theta)
    sub_ik_pvs = fk.fk_elbow_wrist_right_pvs(ik_theta)
    print('Target theta:', tgt_theta)
    print('IK theta:', ik_theta)
    print('Target PVRM:', tgt_pvs)
    print('IK PVRM:', ik_pvs)
    print('Sub target PVS:', sub_tgt_pvs)
    print('Sub IK PVS:', sub_ik_pvs)