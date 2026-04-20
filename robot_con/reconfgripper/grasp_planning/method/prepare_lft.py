import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
from itertools import chain
import motion.probabilistic.rrt_connect as rrtc
import robot_sim.robots.Franka_research3.Franka_research3 as Fr
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rg


class PreGrasp:
    def __init__(self, pos_1, rot_1, pos_2, rot_2, pos_3, rot_3, pos_4, rot_4):
        self.rbt = Fr.Franka_research3()
        self.hnd = rg.Reconfgripper()
        self.lft = rg.Reconfgripper().lft
        self.rgt = rg.Reconfgripper().rgt
        self.main = rg.Reconfgripper().body

        self.rrtc = rrtc.RRTConnect(self.rbt)
        self.path = []
        for idx in range(12):
            setattr(self, f"path_{idx + 1}", [])
        self.cum_lengths = []
        self.path_len = []

        self.count = 0
        self.current_robot_mesh = self.rbt.gen_meshmodel()
        self.objcm_list = [self.rbt.base_stand.lnks[0]['collision_model']]
        self.lft_list = gpa.load_pickle_file('finger_1', '../grasp_info/', 'first_lft_grasps.pickle')
        self.rgt_list = gpa.load_pickle_file('finger_2', '../grasp_info/', 'first_rgt_grasps.pickle')

        # 齐次矩阵
        self.T_rec_cir = self.make_homo(rotmat=rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2),
                                        tvec=np.array([0, .065, -.09398]))
        self.finger_1, self.T_cir_w_1 = self.finger(fin_pos=pos_1, fin_rotmat=rot_1)
        self.finger_2, self.T_cir_w_2 = self.finger(fin_pos=pos_2, fin_rotmat=rot_2)
        self.finger_3, self.T_cir_w_3 = self.finger(fin_pos=pos_3, fin_rotmat=rot_3)
        self.finger_4, self.T_cir_w_4 = self.finger(fin_pos=pos_4, fin_rotmat=rot_4)

        self.T_w_rec_1 = np.linalg.inv(self.T_rec_cir @ self.T_cir_w_1)
        self.T_w_rec_2 = np.linalg.inv(self.T_rec_cir @ self.T_cir_w_2)
        self.T_w_rec_3 = np.linalg.inv(self.T_rec_cir @ self.T_cir_w_3)
        self.T_w_rec_4 = np.linalg.inv(self.T_rec_cir @ self.T_cir_w_4)

    def finger(self, fin_pos, fin_rotmat):
        finger = cm.CollisionModel("../objects/finger_b_2.stl")
        finger.set_pos(fin_pos)
        finger.set_rotmat(fin_rotmat)
        finger.set_rgba([.9, .75, .35, 1])
        homo_cir_w = np.linalg.inv(self.make_homo(rotmat=fin_rotmat, tvec=fin_pos))
        return finger, homo_cir_w

    def put_gripper(self, gri_name, fin_name, jaw_center_pos, jaw_center_rotmat, jaw_width):
        if fin_name == "1":
            w_jaw = self.T_w_rec_1 @ self.make_homo(rotmat=jaw_center_rotmat, tvec=jaw_center_pos)
        elif fin_name == "2":
            w_jaw = self.T_w_rec_2 @ self.make_homo(rotmat=jaw_center_rotmat, tvec=jaw_center_pos)
        elif fin_name == "3":
            w_jaw = self.T_w_rec_3 @ self.make_homo(rotmat=jaw_center_rotmat, tvec=jaw_center_pos)
        elif fin_name == "4":
            w_jaw = self.T_w_rec_4 @ self.make_homo(rotmat=jaw_center_rotmat, tvec=jaw_center_pos)

        if gri_name == "lft":
            self.lft.grip_at_with_jcpose(w_jaw[:3, 3], w_jaw[:3, :3], jaw_width)
            gl_s_pos = self.lft.pos
            gl_s_rotmat = self.lft.rotmat
            g_pos = gl_s_rotmat @ np.array([-.053, .018, -.132]) + gl_s_pos
            g_rotmat = gl_s_rotmat
        elif gri_name == "rgt":
            self.rgt.grip_at_with_jcpose(w_jaw[:3, 3], w_jaw[:3, :3], jaw_width)
            gl_s_pos = self.rgt.pos
            gl_s_rotmat = self.rgt.rotmat
            g_pos = gl_s_rotmat @ np.array([.053, -.018, -.132]) + gl_s_pos
            g_rotmat = gl_s_rotmat
        self.hnd.fix_to(g_pos, g_rotmat)
        return g_pos, g_rotmat

    def make_path(self, pos, rotmat, start_conf):
        conf = self.rbt.tracik(tgt_pos=pos, tgt_rotmat=rotmat)
        if conf is None:
            return None, None
        path = self.rrtc.plan(component_name="arm",
                              start_conf=start_conf,
                              goal_conf=conf,
                              obstacle_list=[],
                              ext_dist=0.05)
        return conf, path

    def update_robot_joints(self, jnt_values):
        try:
            if self.current_robot_mesh is not None:
                self.current_robot_mesh.detach()
            self.rbt.fk(jnt_values=jnt_values)
            new_mesh = self.rbt.gen_meshmodel()
            new_mesh.attach_to(base)
            self.current_robot_mesh = new_mesh
            return True
        except Exception as e:
            print(f"更新机器人时出错: {e}")
            return False

    def update_task(self, task):
        self.path_len = [
            self.path_1, self.path_2, self.path_3, self.path_4, self.path_5,
            self.path_6, self.path_7, self.path_8, self.path_9, self.path_10,
        ]
        total = 0
        for p in self.path_len:
            total += len(p)
            self.cum_lengths.append(total)
        if self.count < len(self.path):
            jnt_values = self.path[self.count]
            if jnt_values is not None:
                self.update_robot_joints(jnt_values)
            # 到达finger_1抓取点
            if self.count == self.cum_lengths[1]:
                self.rbt.hold(hnd_name="hnd", objcm=self.finger_1, jawwidth=0.015)
            # 到达finger_1释放点
            if self.count == self.cum_lengths[3]:
                self.rbt.release(hnd_name="hnd", objcm=self.finger_1, jawwidth=0.029)
            # 到达finger_2抓取点
            if self.count == self.cum_lengths[6]:
                self.rbt.hold(hnd_name="hnd", objcm=self.finger_2, jawwidth=0.015)
            # 到达finger_2释放点
            if self.count == self.cum_lengths[8]:
                self.rbt.release(hnd_name="hnd", objcm=self.finger_2, jawwidth=0.029)
            self.count += 1
            return task.again
        else:
            print("任务完成")
            return task.done

    @staticmethod
    def make_homo(rotmat, tvec):
        homo = np.eye(4)
        homo[:3, :3] = rotmat
        homo[:3, 3] = tvec
        return homo

    @staticmethod
    def split_traj(traj, lengths):
        paths = []
        start = 0
        for length in lengths:
            paths.append(traj[start:start + length])
            start += length
        return paths

    def assign_paths(self, paths):
        for i in range(10):
            setattr(self, f"path_{i + 1}", paths[i])
        self.path = list(chain.from_iterable(paths))


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
    pre = PreGrasp(pos_1=finger_1_pos, rot_1=finger_1_rotmat,
                   pos_2=finger_2_pos, rot_2=finger_2_rotmat,
                   pos_3=finger_3_pos, rot_3=finger_3_rotmat,
                   pos_4=finger_4_pos, rot_4=finger_4_rotmat)
    pre.finger_1.attach_to(base)

    # =========================================step1:grasp_fin1=========================================
    lft_pre_list = []
    lft_rel_list = []
    for i in pre.lft_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = i
        g_pos_1, g_rotmat_1 = pre.put_gripper(gri_name="lft", fin_name="1",
                                              jaw_center_pos=jaw_center_pos, jaw_center_rotmat=jaw_center_rotmat,
                                              jaw_width=jaw_width)

        if not pre.hnd.is_mesh_collided(objcm_list=pre.objcm_list):
            if pre.rbt.tracik(tgt_pos=g_pos_1, tgt_rotmat=g_rotmat_1) is not None:
                g_pos_3, g_rotmat_3 = pre.put_gripper(gri_name="lft", fin_name="3",
                                                      jaw_center_pos=jaw_center_pos, jaw_center_rotmat=jaw_center_rotmat,
                                                      jaw_width=jaw_width)

                if not pre.hnd.is_mesh_collided(objcm_list=pre.objcm_list):
                    if pre.rbt.tracik(tgt_pos=g_pos_3, tgt_rotmat=g_rotmat_3) is not None:
                        lft_pre_list.append([g_pos_1, g_rotmat_1, jaw_width])
                        lft_rel_list.append([g_pos_3, g_rotmat_3, jaw_width])

    # =========================================path_plan=========================================
    start_conf = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, math.pi/4*3])
    path_1 = []
    path_2 = []
    path_3 = []
    path_4 = []
    path_5 = []

    for l_pos_1, l_rotmat_1, _ in lft_pre_list:
        l_pos_1_pre = l_pos_1.copy()
        l_rotmat_1_pre = l_rotmat_1.copy()
        l_pos_1_pre[2] += 0.1
        conf_1, pre.path_1 = pre.make_path(pos=l_pos_1_pre, rotmat=l_rotmat_1_pre, start_conf=start_conf)
        path_1_test = pre.path_1.copy()

        if conf_1 is not None and pre.path_1 is not None:
            conf_2, pre.path_2 = pre.make_path(pos=l_pos_1, rotmat=l_rotmat_1, start_conf=conf_1)

            if conf_2 is not None and pre.path_2 is not None:
                conf_3, pre.path_3 = pre.make_path(pos=l_pos_1_pre, rotmat=l_rotmat_1_pre, start_conf=conf_2)

                if conf_3 is not None and pre.path_3 is not None:
                    for l_pos_2, l_rotmat_2, jawwidth in lft_rel_list:
                        conf_4, pre.path_4 = pre.make_path(pos=l_pos_2, rotmat=l_rotmat_2, start_conf=conf_3)

                        if conf_4 is not None and pre.path_4 is not None:
                            l_pos_2_after = l_pos_2.copy()
                            l_rotmat_2_after = l_rotmat_2.copy()
                            l_pos_2_after[2] += 0.1
                            conf_5, pre.path_5 = pre.make_path(pos=l_pos_2_after, rotmat=l_rotmat_2_after, start_conf=conf_4)

                            if conf_5 is not None and pre.path_5 is not None:
                                path_1.append([conf_1, pre.path_1])
                                path_2.append([conf_2, pre.path_2])
                                path_3.append([conf_3, pre.path_3])
                                path_4.append([conf_4, pre.path_4])
                                path_5.append([conf_5, pre.path_5])
                                # break

    gpa.write_pickle_file('path_1', path_1, 'path_info/', 'path_1.pickle')
    gpa.write_pickle_file('path_2', path_2, 'path_info/', 'path_2.pickle')
    gpa.write_pickle_file('path_3', path_3, 'path_info/', 'path_3.pickle')
    gpa.write_pickle_file('path_4', path_4, 'path_info/', 'path_4.pickle')
    gpa.write_pickle_file('path_5', path_5, 'path_info/', 'path_5.pickle')

    # pre.path.extend(pre.path_1)
    # pre.path.extend(pre.path_2)
    # pre.path.extend(pre.path_3)
    # pre.path.extend(pre.path_4)
    # pre.path.extend(pre.path_5)

    # # =========================================animation=========================================
    # try:
    #     taskMgr.doMethodLater(0.05, pre.update_task, "update")
    #     base.run()
    # except KeyboardInterrupt:
    #     print("\n服务器被用户中断")




