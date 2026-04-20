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
    def __init__(self, pos_1, rot_1, pos_2, rot_2, pos_3, rot_3, pos_4, rot_4, tar_pos, tar_rot):
        # ================================robot, gripper, object================================
        self.rbt = Fr.Franka_research3()
        self.hnd = rg.Reconfgripper()
        self.main = rg.Reconfgripper().body
        self.lft = rg.Reconfgripper().lft
        self.rgt = rg.Reconfgripper().rgt

        self.finger_1 = self.put_object("finger_b_2", pos_1, rot_1, [.9, .75, .35, 1])
        self.finger_2 = self.put_object("finger_b_2", pos_2, rot_2, [.9, .75, .35, 1])
        self.finger_3 = self.put_object("finger_b_2", pos_3, rot_3, [.9, .75, .35, 1])
        self.finger_4 = self.put_object("finger_b_2", pos_4, rot_4, [.9, .75, .35, 1])
        self.target = self.put_object("box", tar_pos, tar_rot, [.9, .75, .35, 1])
        self.table = self.put_object("table", np.array([0, 0, 0]), np.eye(3), [.35, .35, .35, 1])

        self.current_robot_mesh = self.rbt.gen_meshmodel()

        # ===================================generate path===================================
        self.rrtc = rrtc.RRTConnect(self.rbt)
        self.pre_grasp_1 = gpa.load_pickle_file('finger_1', '../grasp_info/pickle', 'pre_finger_1_cd.pickle')
        self.pre_grasp_2 = gpa.load_pickle_file('finger_2', '../grasp_info/pickle', 'pre_finger_2_cd.pickle')
        self.formal_grasp_3 = gpa.load_pickle_file('finger_3', '../grasp_info/pickle', 'formal_finger_3_cd.pickle')
        self.formal_grasp_4 = gpa.load_pickle_file('finger_4', '../grasp_info/pickle', 'formal_finger_4_cd.pickle')
        self.path = []
        self.path_len = []
        self.cum_lengths = []
        self.count = 0

        # ================================collision detection================================
        self.objcm_list = [self.table, self.target]

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

    def finger_manipulation(self, gl_pos, gl_rot, start_conf):
        pre_pos = gl_pos.copy()
        pre_pos[2] += 0.1
        c1, p1 = self.make_path(pre_pos, gl_rot, start_conf)
        if c1 is None or p1 is None:
            return None
        c2, p2 = self.make_path(gl_pos, gl_rot, c1)
        if c2 is None or p2 is None:
            return None
        c3, p3 = self.make_path(pre_pos, gl_rot, c2)
        if c3 is None or p3 is None:
            return None
        return {
            "conf": [c1, c2, c3],
            "path": [p1, p2, p3]
        }

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
        if self.count < len(self.path):
            jnt_values = self.path[self.count]
            if jnt_values is not None:
                self.update_robot_joints(jnt_values)
            # 到达finger_1抓取点
            if self.count == self.cum_lengths[1]:
                self.rbt.hold(hnd_name="hnd", objcm=self.finger_1, jawwidth=0.015)
            # 到达finger_1释放点
            if self.count == self.cum_lengths[4]:
                self.rbt.release(hnd_name="hnd", objcm=self.finger_1, jawwidth=0.029)
            # 到达finger_2抓取点
            if self.count == self.cum_lengths[7]:
                self.rbt.hold(hnd_name="hnd", objcm=self.finger_2, jawwidth=0.015)
            # 到达finger_2释放点
            if self.count == self.cum_lengths[10]:
                self.rbt.release(hnd_name="hnd", objcm=self.finger_2, jawwidth=0.029)
            self.count += 1
            return task.again
        else:
            print("任务完成")
            return task.done

    @staticmethod
    def put_object(name, obj_pos, obj_rot, color):
        obj = cm.CollisionModel(f"../objects/{name}.stl")
        obj.set_pos(obj_pos)
        obj.set_rotmat(obj_rot)
        obj.set_rgba(color)
        return obj

    @staticmethod
    def make_homo(pos, rotmat):
        homo = np.eye(4)
        homo[:3, :3] = rotmat
        homo[:3, 3] = pos
        return homo


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
    target_pos = np.array([-.4, .85, 0])
    target_rot = rm.rotmat_from_axangle([1, 0, 0], math.pi/2)
    pre = PreGrasp(pos_1=finger_1_pos, rot_1=finger_1_rotmat,
                   pos_2=finger_2_pos, rot_2=finger_2_rotmat,
                   pos_3=finger_3_pos, rot_3=finger_3_rotmat,
                   pos_4=finger_4_pos, rot_4=finger_4_rotmat,
                   tar_pos=target_pos, tar_rot=target_rot)
    pre.finger_1.attach_to(base)
    pre.finger_2.attach_to(base)
    # pre.finger_3.attach_to(base)
    # pre.finger_4.attach_to(base)
    pre.table.attach_to(base)
    pre.target.attach_to(base)

    total = 0
    found = False
    for grasp_fin_1_info in pre.pre_grasp_1:
        # =============================把finger_1摆放到合适抓取的位置=============================
        fin_1_info, fin_3_info = grasp_fin_1_info
        _, gl_gri_pos_1, gl_gri_rotmat_1 = fin_1_info
        _, gl_gri_pos_3, gl_gri_rotmat_3 = fin_3_info
        # ==================================抓初始finger_1==================================
        res_1 = pre.finger_manipulation(gl_pos=gl_gri_pos_1, gl_rot=gl_gri_rotmat_1, start_conf=np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, math.pi / 4 * 3]))
        if res_1 is None:
            continue
        path_1, path_2, path_3 = res_1["path"]
        _, _, conf_3 = res_1["conf"]
        # ==============================把finger_1放到finger_3位姿==============================
        res_2 = pre.finger_manipulation(gl_pos=gl_gri_pos_3, gl_rot=gl_gri_rotmat_3, start_conf=conf_3)
        if res_2 is None:
            continue
        path_4, path_5, path_6 = res_2["path"]
        _, _, conf_6 = res_2["conf"]

        for grasp_fin_2_info in pre.pre_grasp_2:
            # =============================把finger_2摆放到合适抓取的位置=============================
            fin_2_info, fin_4_info = grasp_fin_2_info
            _, gl_gri_pos_2, gl_gri_rotmat_2 = fin_2_info
            _, gl_gri_pos_4, gl_gri_rotmat_4 = fin_4_info
            # ==============================抓初始finger_2==============================
            res_3 = pre.finger_manipulation(gl_pos=gl_gri_pos_2, gl_rot=gl_gri_rotmat_2, start_conf=conf_6)
            if res_3 is None:
                continue
            path_7, path_8, path_9 = res_3["path"]
            _, _, conf_9 = res_3["conf"]
            # ==============================把finger_2放到finger_4位姿==============================
            res_4 = pre.finger_manipulation(gl_pos=gl_gri_pos_4, gl_rot=gl_gri_rotmat_4, start_conf=conf_9)
            if res_4 is None:
                continue
            path_10, path_11, path_12 = res_4["path"]
            _, _, conf_12 = res_4["conf"]

            # ===============================总路径===============================
            pre.path = (path_1 + path_2 + path_3 + path_4 + path_5 + path_6 +
                        path_7 + path_8 + path_9 + path_10 + path_11 + path_12)
            pre.path_len = [len(path_1), len(path_2), len(path_3), len(path_4), len(path_5), len(path_6),
                            len(path_7), len(path_8), len(path_9), len(path_10), len(path_11), len(path_12)]
            for p in pre.path_len:
                total += p
                pre.cum_lengths.append(total)
            found = True
            break
        if found:
            break

    # =========================================animation=========================================
    try:
        taskMgr.doMethodLater(0.05, pre.update_task, "update")
        base.run()
    except KeyboardInterrupt:
        print("\n服务器被用户中断")

    base.run()






