import numpy as np 
from fk import FK
from ik import IK
import time

if __name__ == '__main__':
    fk = FK()
    ik = IK()
    data = np.zeros((10,6))

    for i in range(10):
        init_theta = (ik.JOINT_LIMIT_MIN + ik.JOINT_LIMIT_MAX)/2
        init_pvs = fk.fk_fingers_right_pvs(init_theta)
        sub_init_pvs = fk.fk_elbow_wrist_right_pvs(init_theta)
        # tgt_theta = init_theta + np.pi/3*np.ones(17)

        # tgt_pvs = fk.fk_fingers_right_pvs(tgt_theta)
        tgt_pvs = np.array([0.3, 0., 0.1, 0.31, 0.01, 0.1, 0.31, -0.01, 0.1])
        sub_tgt_pvs = np.array([0.2, -0.3, 1.0, 0.3, 0., 0.3])

        start_time = time.perf_counter()
        ik_theta = ik.ik_fingers_simple(init_theta, tgt_pvs, True)
        ik_theta = ik.ik_fingers_prioritized(ik_theta, tgt_pvs, sub_tgt_pvs, True)
        end_time = time.perf_counter()
        ik_pvs = fk.fk_fingers_right_pvs(ik_theta)
        sub_ik_pvs = fk.fk_elbow_wrist_right_pvs(ik_theta)

        # print('Target theta:', tgt_theta)
        # print('IK theta:', ik_theta)
        # print('Target PVRM:', tgt_pvs)
        # print('IK PVRM:', ik_pvs)
        data[i,0] = np.linalg.norm(tgt_pvs[:3] - ik_pvs[:3])*1e3
        data[i,1] = np.linalg.norm(tgt_pvs[3:6] - ik_pvs[3:6])*1e3
        data[i,2] = np.linalg.norm(tgt_pvs[6:9] - ik_pvs[6:9])*1e3
        data[i,3] = np.linalg.norm(sub_tgt_pvs[:3] - sub_ik_pvs[:3])*1e3
        data[i,4] = np.linalg.norm(sub_tgt_pvs[3:6] - sub_ik_pvs[3:6])*1e3
        data[i,5] = (end_time - start_time)*1e3
        print(i+1, end_time - start_time)
    print('Error of thumb [mm]:', np.average(data[:,0]))
    print('Error of index [mm]:', np.average(data[:,1]))
    print('Error of middle [mm]:', np.average(data[:,2]))
    print('Error of elbow [mm]:', np.average(data[:,3]))
    print('Error of wrist [mm]:', np.average(data[:,4]))
    print('Performance time[ms]:', np.average(data[:,5]))
    # print('Sub target PVS:', sub_tgt_pvs)
    # print('Sub IK PVS:', sub_ik_pvs)