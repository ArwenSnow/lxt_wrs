import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.pca.pca as rbt
import visualization.panda.world as wd
import modeling.geometric_model as gm
import time
import motion.probabilistic.rrt_connect as rrtc
import modeling.collision_model as cm
import os
from scipy.spatial.transform import Rotation, Slerp


def make_homo(rotmat, tvec):
    """
    将旋转矩阵和位移向量组合成齐次矩阵形式。
    """
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = tvec
    return homo


def update_robot_joints(jnt_values):
    """
    根据关节角度更新机器人状态
    """
    global current_robot_mesh
    try:
        # 移除旧的机器人模型
        if current_robot_mesh is not None:
            current_robot_mesh.detach()
        # 更新机器人关节角度（前向运动学）
        print(jnt_values)
        robot_s.fk(jnt_values=jnt_values)
        # 生成新的机器人模型
        new_mesh = robot_s.gen_meshmodel()
        new_mesh.attach_to(base)
        current_robot_mesh = new_mesh
        return True

    except Exception as e:
        print(f"更新机器人时出错: {e}")
        return False


if __name__ == '__main__':
    start = time.time()
    this_dir, this_filename = os.path.split(__file__)
    base = wd.World(cam_pos=[-1, 2, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    robot_s = rbt.Pca()

    start_conf = np.array([0, 0, 0, 0, 0, 0, 0])
    rrtc_planner = rrtc.RRTConnect(robot_s)

    box = cm.CollisionModel(os.path.join(this_dir, "meshes", "box.stl"), cdprimit_type="box", expand_radius=.001)
    box_pos = np.array([0.3, 0.2, 0.2])
    box_rot = rm.rotmat_from_axangle([1, 0, 0], math.pi/2)
    # box_rot = np.eye(3)
    box.set_pos(box_pos)
    box.set_rotmat(box_rot)
    box.attach_to(base)

    # gm.gen_frame(box_pos, box_rot).attach_to(base)
    goal_conf = robot_s.ik(component_name='arm', tgt_pos=box_pos, tgt_rotmat=box_rot)
    obstacle_list = []
    grasp_path_1 = rrtc_planner.plan(component_name="arm",
                                     start_conf=start_conf,
                                     goal_conf=goal_conf,
                                     obstacle_list=obstacle_list,
                                     ext_dist=0.1,
                                     max_time=300)

    goal_pos = np.array([0.1, 0.2, 0.3])
    goal_rot = np.eye(3)
    goal_conf_2 = robot_s.ik(component_name='arm', tgt_pos=goal_pos, tgt_rotmat=goal_rot)
    print(goal_conf_2)
    grasp_path_2 = rrtc_planner.plan(component_name="arm",
                                     start_conf=goal_conf,
                                     goal_conf=goal_conf_2,
                                     obstacle_list=obstacle_list,
                                     ext_dist=0.1,
                                     max_time=300)

    grasp_path_3 = rrtc_planner.plan(component_name="arm",
                                     start_conf=goal_conf_2,
                                     goal_conf=start_conf,
                                     obstacle_list=obstacle_list,
                                     ext_dist=0.1,
                                     max_time=300)

    grasp_path = []
    grasp_path.extend(grasp_path_1)
    step_1 = len(grasp_path)-1

    grasp_path.extend(grasp_path_2)
    step_2 = len(grasp_path)-1

    grasp_path.extend(grasp_path_3)

    count = 1
    reversible_counter = 1
    robot_mesh = robot_s.gen_meshmodel()
    current_robot_mesh = robot_mesh

    def update_task(task):
        """
        定时任务：检查并处理网络数据
        """
        global count, reversible_counter
        jnt_values = grasp_path[count]
        if jnt_values is not None:
            # 更新机器人状态
            update_robot_joints(jnt_values)
            # if count == step_1:
            #     robot_s.hnd.jaw_to(.03)
            #     robot_s.hold(hnd_name='hnd', objcm=box)
            # if count == step_2:
            #     robot_s.release(hnd_name='hnd', objcm=box)
        count = count + reversible_counter
        if count == len(grasp_path)-1 or count == -1:
            reversible_counter = reversible_counter*-1
        if count == -1:
            count = 0
        return task.again


    try:
        taskMgr.doMethodLater(0.2, update_task, "update")
        base.run()
    except KeyboardInterrupt:
        print("\n服务器被用户中断")

    base.run()
