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
        self.grasp_1 = gpa.load_pickle_file('finger_1', '../grasp_info/', 'finger_1_cd.pickle')
        self.grasp_2 = gpa.load_pickle_file('finger_2', '../grasp_info/', 'finger_2_cd.pickle')
        self.grasp_3 = gpa.load_pickle_file('finger_3', '../grasp_info/', 'finger_3_cd.pickle')
        self.grasp_4 = gpa.load_pickle_file('finger_4', '../grasp_info/', 'finger_4_cd.pickle')
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
            if self.count == self.path_len[1]:
                self.rbt.hold(hnd_name="hnd", objcm=self.finger_1, jawwidth=0.015)
            # # 到达finger_1释放点
            # if self.count == self.path_len[3]:
            #     self.rbt.release(hnd_name="hnd", objcm=self.finger_1, jawwidth=0.029)
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

    for grasp_fin_1_info in pre.grasp_1:
        jaw_width, gl_gri_pos, gl_gri_rotmat = grasp_fin_1_info

        # 到finger_1上方10厘米
        pre_gri_pos = gl_gri_pos.copy()
        pre_gri_pos[2] += .1
        pre_gri_rot = gl_gri_rotmat.copy()
        conf_1, path_1 = pre.make_path(pos=pre_gri_pos, rotmat=pre_gri_rot,
                                       start_conf=np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, math.pi/4*3]))
        if conf_1 is None or path_1 is None:
            continue

        # 到finger_1抓取位置
        conf_2, path_2 = pre.make_path(pos=gl_gri_pos, rotmat=gl_gri_rotmat,
                                       start_conf=conf_1)
        if conf_2 is None or path_2 is None:
            continue

        # 抓起finger_1到上方10厘米
        conf_3, path_3 = pre.make_path(pos=pre_gri_pos, rotmat=pre_gri_rot,
                                       start_conf=conf_2)
        if conf_3 is None or path_3 is None:
            continue

        pre.path_len = [len(path_1), len(path_2), len(path_3)]
        pre.path.extend(path_1)
        pre.path.extend(path_2)
        pre.path.extend(path_3)
        break

    # =========================================animation=========================================
    try:
        taskMgr.doMethodLater(0.05, pre.update_task, "update")
        base.run()
    except KeyboardInterrupt:
        print("\n服务器被用户中断")

    base.run()






