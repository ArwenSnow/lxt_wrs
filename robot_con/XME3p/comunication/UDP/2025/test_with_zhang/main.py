import socket
import time
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.pca.pca as rbt_1
import robot_sim.robots.xme3p.xme3p as rbt_2
import motion.probabilistic.rrt_connect as rrtc
import visualization.panda.world as wd
import modeling.geometric_model as gm


REMOTE_HOST = "2ef61ee3a199fd96.natapp.cc"
REMOTE_PORT = 20257
SEND_HZ = 10
DT = 1.0 / SEND_HZ
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def make_homo(rotmat, tvec):
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


def main(homomat_list):
    print(f"UDP Pose Sender -> {REMOTE_HOST}:{REMOTE_PORT} @ {SEND_HZ}Hz")
    print("发送固定4×4齐次变换矩阵（16个float64，128字节）")
    seq = 0
    while True:
        for homo in homomat_list:
            payload = homo.tobytes()
            sock.sendto(payload, (REMOTE_HOST, REMOTE_PORT))

        # 控制台打印（每0.5秒左右一次）
            if seq % int(SEND_HZ / 2) == 0:
                print(f"\nseq={seq}")
                print(f"固定齐次变换矩阵：")
                print(f"{homo.round(6)}")
                print(f"发送字节数：{len(payload)}")

            seq += 1
            # 控制发送频率
            time.sleep(0.1)


if __name__ == "__main__":
    base = wd.World(cam_pos=[-3, 2, 1], lookat_pos=[1, 0, 0])
    gm.gen_frame().attach_to(base)

    robot_1 = rbt_1.Pca()
    # robot_1.gen_meshmodel().attach_to(base)
    current_robot_mesh_1 = robot_1.gen_meshmodel()

    robot_2 = rbt_2.XME3P()
    rbt_2_pos = np.array([.50682, .6, 0])
    rbt_2_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi)
    robot_2.fix_to(rbt_2_pos, rbt_2_rot)

    rrtc_1 = rrtc.RRTConnect(robot_1)
    rrtc_2 = rrtc.RRTConnect(robot_2)

    start_conf = np.array([0, .349066, 0, 1.0472, 0, 1.5708, 0])
    robot_2.fk('arm', start_conf)
    # robot_2.gen_meshmodel().attach_to(base)
    current_robot_mesh_2 = robot_2.gen_meshmodel()

    p, r = robot_2.get_gl_tcp('arm')
    goal_p = p + np.array([.1, .2, .1])
    goal_homo = make_homo(r, goal_p)

    goal_conf = robot_2.ik('arm', goal_p, r)
    robot_2.fk('arm', goal_conf)
    # robot_2.gen_meshmodel().attach_to(base)

    obstacle_list = []
    p_detla = np.array([.2, .2, 0])

    b = robot_1.ik('arm', p-p_detla, r)
    if b is not None:
        print("遥感机械臂能解起点ik")
    else:
        print("不能")

    c = robot_1.ik('arm', goal_p-p_detla, r)
    if c is not None:
        print("遥感机械臂能解终点ik")
    else:
        print("不能")

    path_1 = rrtc_1.plan(component_name="arm",
                         start_conf=robot_1.ik('arm', p-p_detla, r),
                         goal_conf=robot_1.ik('arm', goal_p-p_detla, r),
                         obstacle_list=obstacle_list,
                         ext_dist=0.002,
                         max_time=300)

    path_2 = rrtc_2.plan(component_name="arm",
                         start_conf=start_conf,
                         goal_conf=goal_conf,
                         obstacle_list=obstacle_list,
                         ext_dist=0.002,
                         max_time=300)

    count_1 = 1
    count_2 = 1
    reversible_counter_1 = 1
    reversible_counter_2 = 1

    def update_task_1(task):
        """
        定时任务：检查并处理网络数据
        """
        global count_1, reversible_counter_1
        jnt_values = path_1[count_1]
        if jnt_values is not None:
            # 更新机器人状态
            update_robot_joints_1(robot_1, jnt_values)
        count_1 = count_1 + reversible_counter_1
        if count_1 == len(path_1)-1 or count_1 == -1:
            reversible_counter_1 = reversible_counter_1*-1
        if count_1 == -1:
            count_1 = 0
        return task.again


    def update_task_2(task):
        """
        定时任务：检查并处理网络数据
        """
        global count_2, reversible_counter_2
        jnt_values = path_2[count_2]
        if jnt_values is not None:
            # 更新机器人状态
            update_robot_joints_2(robot_2, jnt_values)
        count_2 = count_2 + reversible_counter_2
        if count_2 == len(path_2)-1 or count_2 == -1:
            reversible_counter_2 = reversible_counter_2*-1
        if count_2 == -1:
            count_2 = 0
        return task.again

    try:
        taskMgr.doMethodLater(0.005, update_task_1, "update")
        taskMgr.doMethodLater(0.005, update_task_2, "update")
        base.run()
    except KeyboardInterrupt:
        print("\n服务器被用户中断")

    # path_list = []
    # a = np.asarray(goal_homo, dtype=np.float64)
    # path_list.append(a)
    # main(path_list)






