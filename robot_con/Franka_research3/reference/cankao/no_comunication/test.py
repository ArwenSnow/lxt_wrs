import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.pca.pca as rbt_1
import robot_sim.robots.xme3p.xme3p as rbt_2
import modeling.collision_model as cm
import time
import os


def make_homo(rotmat, tvec):
    """
    将旋转矩阵和位移向量组合成齐次矩阵形式。
    """
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = tvec
    return homo


def update_robot_joints_1(robot_s, jnt_values):
    """
    根据关节角度更新机器人状态。
    """
    global current_robot_mesh_1
    try:
        # 移除旧的机器人模型
        if current_robot_mesh_1 is not None:
            current_robot_mesh_1.detach()
        # 更新机器人关节角度（前向运动学）
        # print(jnt_values)
        robot_s.fk(jnt_values=jnt_values)
        # 生成新的机器人模型
        new_mesh = robot_s.gen_meshmodel(toggle_tcpcs=True)
        new_mesh.attach_to(base)
        current_robot_mesh_1 = new_mesh
        return True

    except Exception as e:
        print(f"更新机器人时出错: {e}")
        return False


def update_robot_joints_2(robot_s, jnt_values):
    """
    根据关节角度更新机器人状态。
    """
    global current_robot_mesh_2
    try:
        # 移除旧的机器人模型
        if current_robot_mesh_2 is not None:
            current_robot_mesh_2.detach()
        # 更新机器人关节角度（前向运动学）
        # print(jnt_values)
        robot_s.fk(jnt_values=jnt_values)
        # 生成新的机器人模型
        new_mesh = robot_s.gen_meshmodel(toggle_tcpcs=True)
        new_mesh.attach_to(base)
        current_robot_mesh_2 = new_mesh
        return True

    except Exception as e:
        print(f"更新机器人时出错: {e}")
        return False


def make_circle_with_rot(center, initial_rotmat):
    radius = 0.3
    num_points = 50
    path_positions = []
    path_rotations = []

    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        # 圆周位置
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        z = center[2]
        path_positions.append(np.array([x, y, z]))

        # 绕z轴
        rot = rm.rotmat_from_axangle([0, 0, 1], theta) @ initial_rotmat
        path_rotations.append(rot)

    return path_positions, path_rotations


if __name__ == '__main__':
    start = time.time()
    this_dir, this_filename = os.path.split(__file__)
    base = wd.World(cam_pos=[-3, 2, 1], lookat_pos=[1, 0, 0])
    gm.gen_frame().attach_to(base)

    # rbt_s
    robot_1 = rbt_1.Pca()
    pos_11, rot_11 = robot_1.get_gl_tcp('arm')
    T_1 = make_homo(rot_11, pos_11)

    robot_2 = rbt_2.XME3P()
    rbt_2_pos = np.array([.50682, .6, 0])
    rbt_2_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi)
    robot_2.fix_to(rbt_2_pos, rbt_2_rot)
    pos_22, rot_22 = robot_2.get_gl_tcp('arm')
    T_2 = make_homo(rot_22, pos_22)

    # box
    box = cm.CollisionModel(os.path.join(this_dir, "meshes", "box.STL"), cdprimit_type="box", expand_radius=.001)

    # path_1
    path_init = []
    path_1 = []
    path_11 = []
    pos_1, rot_1 = [], []
    pos_init, rot_init = make_circle_with_rot(np.array([.3, .3, .6]), np.eye(3))
    for pos, rot in zip(pos_init, rot_init):
        conf = robot_1.ik(component_name='arm', tgt_pos=pos, tgt_rotmat=rot)
        if conf is not None:
            path_init.append(conf)
            pos_1.append(pos)
            rot_1.append(rot)

    # path_2
    path_2 = []
    path_22 = []
    pos_2, rot_2 = [], []

    T_z = np.eye(4)
    T_z[:3, :3] = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi)

    for p, r in zip(pos_1, rot_1):
        pos_2.append(p + np.array([.50682, .3, 0]))
        rot_2.append(r)

    i = 1
    for pos, rot in zip(pos_2, rot_2):
        conf = robot_2.ik(component_name='arm', tgt_pos=pos, tgt_rotmat=rot)
        if conf is not None:
            path_2.append(conf)
            path_1.append(path_init[i])
            i += 1

    for i, _ in enumerate(path_1):
        if 0 <= i <= 6:
            path_11.append(path_1[i])
            path_22.append(path_2[i])

    if len(path_11) == len(path_22):
        print("两条路径途径点数量一致，点数为", len(path_11))
    else:
        print(len(path_11), len(path_22))

    count_1 = 1
    count_2 = 1
    reversible_counter_1 = 1
    reversible_counter_2 = 1
    robot_mesh_1 = robot_1.gen_meshmodel(toggle_tcpcs=True)
    robot_mesh_2 = robot_2.gen_meshmodel(toggle_tcpcs=True, rgba=[.7, .7, .7, .5])

    current_robot_mesh_1 = robot_mesh_1
    current_robot_mesh_2 = robot_mesh_2

    def update_task_1(task):
        """
        定时任务：检查并处理网络数据
        """
        global count_1, reversible_counter_1
        jnt_values = path_11[count_1]
        if jnt_values is not None:
            # 更新机器人状态
            update_robot_joints_1(robot_1, jnt_values)
        count_1 = count_1 + reversible_counter_1
        if count_1 == len(path_11)-1 or count_1 == -1:
            reversible_counter_1 = reversible_counter_1*-1
        if count_1 == -1:
            count_1 = 0
        return task.again

    def update_task_2(task):
        """
        定时任务：检查并处理网络数据
        """
        global count_2, reversible_counter_2
        jnt_values = path_22[count_2]
        if jnt_values is not None:
            # 更新机器人状态
            update_robot_joints_2(robot_2, jnt_values)
        count_2 = count_2 + reversible_counter_2
        if count_2 == len(path_22)-1 or count_2 == -1:
            reversible_counter_2 = reversible_counter_2*-1
        if count_2 == -1:
            count_2 = 0
        return task.again

    try:
        taskMgr.doMethodLater(0.2, update_task_1, "update")
        taskMgr.doMethodLater(0.2, update_task_2, "update")
        base.run()
    except KeyboardInterrupt:
        print("\n服务器被用户中断")