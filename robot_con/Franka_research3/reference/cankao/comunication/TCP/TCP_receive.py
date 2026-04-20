import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.pca.pca as rbt_1
import robot_sim.robots.xme3p.xme3p as rbt_2
import motion.probabilistic.rrt_connect as rrtc
import modeling.collision_model as cm
import time
import os
import socket
import json


# 创建服务器
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', 50000))
server.listen(1)
print("服务器已启动，等待客户端连接...")
conn, addr = server.accept()
print(f"客户端已连接: {addr}")

angles_list = []
try:
    while True:
        data = conn.recv(1024)
        if not data:
            break
        msg = data.decode()
        try:
            # 解析 JSON
            msg_dict = json.loads(msg)
            send_time = msg_dict["time"]           # 发送端时间
            angles = msg_dict["angles"]            # 角度数据
            receive_time = time.time()             # 接收端时间
            delay = receive_time - send_time       # 延迟秒数
            angles_list.append(np.array(angles))
            print(f"收到角度: {angles}, 延迟: {delay*1000:.2f} ms")  # ms更直观
        except (json.JSONDecodeError, KeyError):
            print("无法解析为列表或缺少字段")
finally:
    conn.close()
    server.close()

# # 创建服务器
# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind(('127.0.0.1', 50000))
# server.listen(1)
# print("服务器已启动，等待客户端连接...")
# conn, addr = server.accept()
# print(f"客户端已连接: {addr}")

# angles_list = []
# try:
#     while True:
#         data = conn.recv(1024)
#         if not data:
#             break
#         msg = data.decode()
#         try:
#             # 解析 JSON
#             msg_parsed = json.loads(msg)
#
#             # 检查是否为字典
#             if isinstance(msg_parsed, dict) and "time" in msg_parsed and "angles" in msg_parsed:
#                 send_time = msg_parsed["time"]           # 发送端时间
#                 angles = msg_parsed["angles"]            # 角度数据
#                 receive_time = time.time()               # 接收端时间
#                 delay = receive_time - send_time         # 延迟秒数
#                 angles_list.append(np.array(angles))
#                 print(f"收到角度: {angles}, 延迟: {delay*1000:.2f} ms")  # ms更直观
#             else:
#                 print("收到数据:", msg_parsed)
#
#         except json.JSONDecodeError:
#             print("无法解析为 JSON:", msg)
#         except Exception as e:
#             print(f"处理数据时出错: {e}")
#
# finally:
#     conn.close()
#     server.close()


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


start = time.time()
this_dir, this_filename = os.path.split(__file__)
base = wd.World(cam_pos=[-3, 2, 1], lookat_pos=[1, 0, 0])
gm.gen_frame().attach_to(base)

# rbt_s
robot_1 = rbt_1.Pca()

count_1 = 1
reversible_counter_1 = 1
robot_mesh_1 = robot_1.gen_meshmodel(toggle_tcpcs=True)
current_robot_mesh_1 = robot_mesh_1


def update_task_1(task):
    """
    定时任务：检查并处理网络数据
    """
    global count_1, reversible_counter_1
    jnt_values = angles_list[count_1]
    if jnt_values is not None:
        # 更新机器人状态
        update_robot_joints_1(robot_1, jnt_values)
    count_1 = count_1 + reversible_counter_1
    if count_1 == len(angles_list) - 1 or count_1 == -1:
        reversible_counter_1 = reversible_counter_1 * -1
    if count_1 == -1:
        count_1 = 0
    return task.again


try:
    taskMgr.doMethodLater(0.005, update_task_1, "update")
    base.run()
except KeyboardInterrupt:
    print("\n服务器被用户中断")



