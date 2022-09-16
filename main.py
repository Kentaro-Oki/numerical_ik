import numpy as np 
from fk import FK
from ik import IK

if __name__ == '__main__':
    fk = FK()
    ik = IK()
    init_theta = (ik.JOINT_LIMIT_MIN + ik.JOINT_LIMIT_MAX)/2
    tgt_theta = init_theta + np.pi/6*np.ones(17)
    tgt_pvrm = fk.fk_arm_right_pvrm(tgt_theta)
    ik_theta = ik.ik_arm_simple(init_theta, tgt_pvrm, True)
    tgt_pvrv = fk.fk_arm_right_pvrv(tgt_theta)
    ik_pvrv = fk.fk_arm_right_pvrv(ik_theta)
    print('Target theta:', tgt_theta)
    print('IK theta:', ik_theta)
    print('Target PVRV:', tgt_pvrv)
    print('IK PVRV:', ik_pvrv)