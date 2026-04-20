import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import grasping.planning.antipodal as gpa
import modeling.collision_model as cm
import modeling.geometric_model as gm
import robot_con.reconfgripper.grasp_planning_old.method_2.prepare_lft as pp


class FormalGrasp(pp.PreGrasp):
    def __init__(self, pos_1, rot_1, pos_2, rot_2, pos_3, rot_3, pos_4, rot_4, pos_t, rot_t):
        super().__init__(pos_1=pos_1, rot_1=rot_1, pos_2=pos_2, rot_2=rot_2,
                         pos_3=pos_3, rot_3=rot_3, pos_4=pos_4, rot_4=rot_4)
        self.formal_lft_list = gpa.load_pickle_file('finger_1', '../grasp_info/', 'lft_grasps.pickle')
        self.formal_rgt_list = gpa.load_pickle_file('finger_2', '../grasp_info/', 'rgt_grasps.pickle')

        # grasp fingers
        self.pre_path_info = gpa.load_pickle_file('path_pre', 'path_info/', 'path_pre.pickle')
        self.pre_traj = self.pre_path_info["trajectory"]
        self.pre_lens = self.pre_path_info["lengths"]
        self.conf_10 = self.pre_path_info["conf_10"]
        self.pre_path = self.split_traj(traj=self.pre_traj, lengths=self.pre_lens)
        self.assign_paths(self.pre_path)

        # target
        self.target = self.set_target(pos_tar=pos_t, rot_tar=rot_t)
        self.target_for_plan = self.set_target(pos_tar=np.array([0, 0, 0]), rot_tar=np.eye(3))
        self.T_w_tar = self.make_homo(rotmat=rot_t, tvec=pos_t)
        self.T_w_jaw_new = self.make_homo(rotmat=rot_3, tvec=(pos_3 + pos_4)/2 + np.array([0, 0.12549, 0]))
        self.T_gri_w = []

        # collision detection
        self.table = cm.CollisionModel("../objects/table.stl")
        self.new_objcm_list = [self.rbt.base_stand.lnks[0]['collision_model'], self.target]

    @staticmethod
    def set_target(pos_tar, rot_tar):
        target = cm.CollisionModel("../objects/box.stl")
        target.set_pos(pos_tar)
        target.set_rotmat(rot_tar)
        target.set_rgba([.9, .75, .35, 1])
        return target

    def put_gripper_fin(self, gri_name, fin_name, jaw_center_pos, jaw_center_rotmat, jaw_width):
        if fin_name == "1":
            w_jaw = np.linalg.inv(self.T_cir_w_1) @ self.make_homo(rotmat=jaw_center_rotmat, tvec=jaw_center_pos)
        elif fin_name == "2":
            w_jaw = np.linalg.inv(self.T_cir_w_2) @ self.make_homo(rotmat=jaw_center_rotmat, tvec=jaw_center_pos)
        elif fin_name == "3":
            w_jaw = np.linalg.inv(self.T_cir_w_3) @ self.make_homo(rotmat=jaw_center_rotmat, tvec=jaw_center_pos)
        elif fin_name == "4":
            w_jaw = np.linalg.inv(self.T_cir_w_4) @ self.make_homo(rotmat=jaw_center_rotmat, tvec=jaw_center_pos)
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

    def update_task_formal(self, task):
        self.path_len = [
            self.path_1, self.path_2, self.path_3, self.path_4, self.path_5,
            self.path_6, self.path_7, self.path_8, self.path_9, self.path_10,
            self.path_11, self.path_12, self.path_13, self.path_14
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
            # 到达一对手指抓取点
            if self.count == self.cum_lengths[11]:
                self.rbt.hold(hnd_name="hnd", objcm=self.finger_1, jawwidth=0.015)
                self.rbt.hold(hnd_name="hnd", objcm=self.finger_2, jawwidth=0.015)
                self.rbt.hnd.rg_jaw_to(jaw_width=0.015)
            # # 成功重构手指后
            # if self.count == self.cum_lengths[12]:
            #     self.rbt.hnd.mg_open()
            self.count += 1
            return task.again
        else:
            print("任务完成")
            return task.done


if __name__ == '__main__':
    base = wd.World(cam_pos=[-2, 4, 1.5], lookat_pos=[0, 0, 0])

    # =========================================obj_pose=========================================
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
    tar_pos = np.array([-.4, .85, 0])
    tar_rot = rm.rotmat_from_axangle([1, 0, 0], math.pi/2)

    formal = FormalGrasp(pos_1=finger_1_pos, rot_1=finger_1_rotmat,
                         pos_2=finger_2_pos, rot_2=finger_2_rotmat,
                         pos_3=finger_3_pos, rot_3=finger_3_rotmat,
                         pos_4=finger_4_pos, rot_4=finger_4_rotmat,
                         pos_t=tar_pos, rot_t=tar_rot)
    # formal.rbt.gen_meshmodel(rgba=[0, 0, 0, .1]).attach_to(base)
    # formal.finger_1.attach_to(base)
    # formal.finger_2.attach_to(base)
    formal.target.attach_to(base)
    formal.table.set_rgba([0, 0, 0, .1])
    # formal.table.attach_to(base)

    # =========================================step3:grasp_fins=========================================
    fins_list = []
    for i in formal.formal_lft_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = i
        g_pos_3, g_rotmat_3 = formal.put_gripper_fin(gri_name="lft", fin_name="4",
                                                     jaw_center_pos=jaw_center_pos, jaw_center_rotmat=jaw_center_rotmat,
                                                     jaw_width=jaw_width)

        if not formal.hnd.is_mesh_collided(objcm_list=formal.objcm_list):
            if formal.rbt.tracik(tgt_pos=g_pos_3, tgt_rotmat=g_rotmat_3) is not None:
                fins_list.append([g_pos_3, g_rotmat_3, jaw_width])

    # =========================================path_plan============================================
    start_conf = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, math.pi/4*3])
    path_11 = []
    path_12 = []
    path_13 = []

    for fins_pos, fins_rotmat, _ in fins_list:
        fins_pos_pre = fins_pos.copy()
        fins_rotmat_pre = fins_rotmat.copy()
        fins_pos_pre[2] += 0.1
        conf_11, formal.path_11 = formal.make_path(pos=fins_pos_pre, rotmat=fins_rotmat_pre,
                                                   start_conf=formal.conf_10)

        if conf_11 is not None and formal.path_11 is not None:
            conf_12, formal.path_12 = formal.make_path(pos=fins_pos, rotmat=fins_rotmat,
                                                       start_conf=conf_11)

            if conf_12 is not None and formal.path_12 is not None:
                fins_pos_aft = fins_pos.copy()
                fins_rotmat_aft = fins_rotmat.copy()
                fins_pos_aft[2] += 0.2
                conf_13, formal.path_13 = formal.make_path(pos=fins_pos_aft, rotmat=fins_rotmat_aft,
                                                           start_conf=conf_12)
                # 为改变夹爪抓取中心做准备
                formal.rbt.fk(jnt_values=conf_12)
                formal.T_gri_w = np.linalg.inv(formal.make_homo(formal.rbt.hnd.rotmat, formal.rbt.hnd.pos))
                T_gri_fin_3 = formal.T_gri_w @ formal.make_homo(finger_3_rotmat, finger_3_pos)
                T_gri_fin_4 = formal.T_gri_w @ formal.make_homo(finger_4_rotmat, finger_4_pos)
                break

    # =========================================step4:grasp_obj=========================================
    # 改变夹爪的抓取中心，T_gri_jaw = T_gri_w @ T_w_jaw
    formal.main.jawwidth_rng = [0.0, .193]
    formal.main.jaw_center_pos = formal.T_gri_w[:3, :3] @ formal.T_w_jaw_new[:3, 3] + formal.T_gri_w[:3, 3]
    formal.main.jaw_center_rotmat = formal.T_gri_w[:3, :3] @ formal.T_w_jaw_new[:3, :3]

    object_grasp_info_list = gpa.plan_grasps(formal.main, formal.target_for_plan,
                                             angle_between_contact_normals=math.radians(160),
                                             openning_direction='loc_x',
                                             max_samples=5, min_dist_between_sampled_contact_points=.005,
                                             contact_offset=.001)

    for i in object_grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = i
        T_w_jaw = formal.T_w_tar @ formal.make_homo(rotmat=jaw_center_rotmat, tvec=jaw_center_pos)

        formal.main.grip_at_with_jcpose(T_w_jaw[:3, 3], T_w_jaw[:3, :3], jaw_width)
        formal.rbt.hnd.fix_to(formal.main.pos, formal.main.rotmat)

        T_w_fin_3 = formal.make_homo(formal.main.rotmat, formal.main.pos) @ T_gri_fin_3
        formal.finger_3.set_pos(T_w_fin_3[:3, 3])
        formal.finger_3.set_rotmat(T_w_fin_3[:3, :3])

        T_w_fin_4 = formal.make_homo(formal.main.rotmat, formal.main.pos) @ T_gri_fin_4
        formal.finger_4.set_pos(T_w_fin_4[:3, 3])
        formal.finger_4.set_rotmat(T_w_fin_4[:3, :3])

        conf_14 = formal.rbt.tracik(tgt_pos=formal.main.pos, tgt_rotmat=formal.main.rotmat)
        if conf_14 is not None:
            formal.rbt.fk(jnt_values=conf_14)
            formal.hnd.fix_to(formal.main.pos, formal.main.rotmat)

            if not formal.hnd.is_mesh_collided(objcm_list=formal.new_objcm_list):
                formal.finger_3.attach_to(base)
                # gm.gen_frame(pos=T_w_fin_3[:3, 3], rotmat=T_w_fin_3[:3, :3]).attach_to(base)
                formal.finger_4.attach_to(base)
                # gm.gen_frame(pos=T_w_fin_4[:3, 3], rotmat=T_w_fin_4[:3, :3]).attach_to(base)
                formal.hnd.gen_meshmodel().attach_to(base)
                # gm.gen_frame(pos=tar_pos, rotmat=tar_rot).attach_to(base)
                break
            #
            # if formal.hnd.is_mesh_collided(objcm_list=formal.new_objcm_list):
            #     # formal.finger_3.attach_to(base)
            #     # gm.gen_frame(pos=T_w_fin_3[:3, 3], rotmat=T_w_fin_3[:3, :3]).attach_to(base)
            #     # formal.finger_4.attach_to(base)
            #     # gm.gen_frame(pos=T_w_fin_4[:3, 3], rotmat=T_w_fin_4[:3, :3]).attach_to(base)
            #     formal.hnd.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)
            #     # gm.gen_frame(pos=tar_pos, rotmat=tar_rot).attach_to(base)

    #
    # formal.path_14 = formal.rrtc.plan(component_name="arm",
    #                                   start_conf=conf_13,
    #                                   goal_conf=conf_14,
    #                                   obstacle_list=[],
    #                                   ext_dist=0.05)
    #
    # # =========================================animation=========================================
    # formal.path.extend(formal.path_11)
    # formal.path.extend(formal.path_12)
    # formal.path.extend(formal.path_13)
    # formal.path.extend(formal.path_14)
    # try:
    #     taskMgr.doMethodLater(0.05, formal.update_task_formal, "update")
    #     base.run()
    # except KeyboardInterrupt:
    #     print("\n服务器被用户中断")
    #
    base.run()




