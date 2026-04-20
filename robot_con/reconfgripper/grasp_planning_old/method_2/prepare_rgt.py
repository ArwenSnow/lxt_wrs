import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import grasping.planning.antipodal as gpa
import robot_con.reconfgripper.grasp_planning.method_2.prepare_lft as pp

if __name__ == '__main__':
    base = wd.World(cam_pos=[-2, 4, 1.5], lookat_pos=[0, 0, 0])

    # =========================================finger_pose=========================================
    finger_1_pos = np.array([-.3, .5, .015])
    finger_1_rotmat = (rm.rotmat_from_axangle([0, 0, 1], math.pi / 2 * 1.2)
                       @ rm.rotmat_from_axangle([1, 0, 0], math.pi / 180 * 87.3))
    finger_2_pos = np.array([.3, .5, .015])
    finger_2_rotmat = (rm.rotmat_from_axangle([1, 0, 0], -math.pi / 180 * 87.3)
                       @ rm.rotmat_from_axangle([0, 1, 0], math.pi / 2 * 0.5))
    finger_3_pos = np.array([-.0265, .75, .015])
    finger_3_rotmat = (rm.rotmat_from_axangle([1, 0, 0], -math.pi / 180 * 87.3)
                       @ rm.rotmat_from_axangle([0, 0, 1], math.pi))
    finger_4_pos = np.array([.0265, .75, .015])
    finger_4_rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 180 * 87.3)

    pre = pp.PreGrasp(pos_1=finger_1_pos, rot_1=finger_1_rotmat,
                      pos_2=finger_2_pos, rot_2=finger_2_rotmat,
                      pos_3=finger_3_pos, rot_3=finger_3_rotmat,
                      pos_4=finger_4_pos, rot_4=finger_4_rotmat)
    pre.finger_1.attach_to(base)
    pre.finger_2.attach_to(base)

    # =========================================step2:grasp_fin2=========================================
    rgt_pre_list = []
    rgt_rel_list = []
    for i in pre.lft_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = i
        g_pos_2, g_rotmat_2 = pre.put_gripper(gri_name="lft", fin_name="2",
                                              jaw_center_pos=jaw_center_pos, jaw_center_rotmat=jaw_center_rotmat,
                                              jaw_width=jaw_width)

        if not pre.hnd.is_mesh_collided(objcm_list=pre.objcm_list):
            if pre.rbt.tracik(tgt_pos=g_pos_2, tgt_rotmat=g_rotmat_2) is not None:
                g_pos_4, g_rotmat_4 = pre.put_gripper(gri_name="lft", fin_name="4",
                                                      jaw_center_pos=jaw_center_pos, jaw_center_rotmat=jaw_center_rotmat,
                                                      jaw_width=jaw_width)

                if not pre.hnd.is_mesh_collided(objcm_list=pre.objcm_list):
                    if pre.rbt.tracik(tgt_pos=g_pos_4, tgt_rotmat=g_rotmat_4) is not None:
                        rgt_pre_list.append([g_pos_2, g_rotmat_2, jaw_width])
                        rgt_rel_list.append([g_pos_4, g_rotmat_4, jaw_width])

    # =========================================path_plan=========================================
    path_1_info = gpa.load_pickle_file('path_1', 'path_info/', 'path_1.pickle')
    path_2_info = gpa.load_pickle_file('path_2', 'path_info/', 'path_2.pickle')
    path_3_info = gpa.load_pickle_file('path_3', 'path_info/', 'path_3.pickle')
    path_4_info = gpa.load_pickle_file('path_4', 'path_info/', 'path_4.pickle')
    path_5_info = gpa.load_pickle_file('path_5', 'path_info/', 'path_5.pickle')

    num = 0

    found = False
    idx_5 = 0
    conf_6_list = []
    path_5 = []
    path_6 = []
    path_7 = []
    path_8 = []
    path_9 = []
    path_10 = []
    for idx, (r_pos_1, r_rotmat_1, jawwidth) in enumerate(rgt_pre_list):
        r_pos_1_pre = r_pos_1.copy()
        r_rotmat_1_pre = r_rotmat_1.copy()
        r_pos_1_pre[2] += 0.1
        conf_6 = pre.rbt.tracik(tgt_pos=r_pos_1_pre, tgt_rotmat=r_rotmat_1_pre)
        conf_6_list.append(conf_6)

        if conf_6 is not None:
            conf_7, pre.path_7 = pre.make_path(pos=r_pos_1, rotmat=r_rotmat_1, start_conf=conf_6)

            if conf_7 is not None and pre.path_7 is not None:
                conf_8, pre.path_8 = pre.make_path(pos=r_pos_1_pre, rotmat=r_rotmat_1_pre, start_conf=conf_7)

                if conf_8 is not None and pre.path_8 is not None:
                    for r_pos_2, r_rotmat_2, _ in rgt_rel_list:
                        conf_9, pre.path_9 = pre.make_path(pos=r_pos_2, rotmat=r_rotmat_2, start_conf=conf_8)

                        if conf_9 is not None and pre.path_9 is not None:
                            r_pos_2_after = r_pos_2.copy()
                            r_rotmat_2_after = r_rotmat_2.copy()
                            r_pos_2_after[2] += 0.1
                            conf_10, pre.path_10 = pre.make_path(pos=r_pos_2_after, rotmat=r_rotmat_2_after, start_conf=conf_9)

                            if conf_10 is not None and pre.path_10 is not None:
                                found = True
                                idx_5 = idx
                                break
        if found:
            break

    idx_1 = 0
    for idx, path_5_list in enumerate(path_5_info):
        conf_5, pre.path_5 = path_5_list

        pre.path_6 = pre.rrtc.plan(component_name="arm",
                                   start_conf=conf_5,
                                   goal_conf=conf_6_list[idx_5],
                                   obstacle_list=[],
                                   ext_dist=0.05)

        if pre.path_6 is not None:
            idx_1 = idx
            break

    pre.path_1 = path_1_info[idx_1][1]
    pre.path_2 = path_2_info[idx_1][1]
    pre.path_3 = path_3_info[idx_1][1]
    pre.path_4 = path_4_info[idx_1][1]

    pre.path.extend(pre.path_1)
    pre.path.extend(pre.path_2)
    pre.path.extend(pre.path_3)
    pre.path.extend(pre.path_4)
    pre.path.extend(pre.path_5)
    pre.path.extend(pre.path_6)
    pre.path.extend(pre.path_7)
    pre.path.extend(pre.path_8)
    pre.path.extend(pre.path_9)
    pre.path.extend(pre.path_10)

    path_pre = {
        "lengths": [
            len(pre.path_1), len(pre.path_2), len(pre.path_3), len(pre.path_4),
            len(pre.path_5), len(pre.path_6), len(pre.path_7),
            len(pre.path_8), len(pre.path_9), len(pre.path_10)
        ],
        "trajectory": pre.path,
        "conf_10": conf_10
    }
    gpa.write_pickle_file('path_pre', path_pre, 'path_info/', 'path_pre.pickle')

    # =========================================animation=========================================
    try:
        taskMgr.doMethodLater(0.08, pre.update_task, "update")
        base.run()
    except KeyboardInterrupt:
        print("\n服务器被用户中断")


