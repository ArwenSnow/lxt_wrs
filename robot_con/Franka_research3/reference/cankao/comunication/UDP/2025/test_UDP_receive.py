import time
import socket
import json
import numpy as np
import threading
import os
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.pca.pca as rbt_1


def make_homo(rotmat, tvec):
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = tvec
    return homo


def update_robot_joints(robot, jnt_values):
    global current_robot_mesh
    try:
        if current_robot_mesh is not None:
            current_robot_mesh.detach()

        robot.fk(jnt_values=jnt_values)
        new_mesh = robot.gen_meshmodel(toggle_tcpcs=True)
        new_mesh.attach_to(base)
        current_robot_mesh = new_mesh
    except Exception as e:
        print(f"更新机器人失败: {e}")


UDP_IP = '0.0.0.0'
UDP_PORT = 11111
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((UDP_IP, UDP_PORT))
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)    # 接收缓冲区64kb
print("UDP 服务器已启动，等待数据...")

latest_angles = None
prev_angles = None
latest_time = None
data_lock = threading.Lock()


def udp_receiver():
    global latest_angles, latest_time
    while True:                                                         # UDP需要循环执行
        try:
            data, addr = server.recvfrom(1024)
            msg = json.loads(data.decode())                             # bytes → str → dict

            # send_time = msg["time"]
            # angles = np.array(msg["angles"])
            angles = np.array(msg)

            # recv_time = time.time()
            # delay = (recv_time - send_time) * 1000.0

            with data_lock:                                             # 改数据时，主线程不能来读
                latest_angles = angles
                # latest_time = delay

        except Exception as e:
            print(f"UDP 接收错误: {e}")


recv_thread = threading.Thread(target=udp_receiver, daemon=True)       # 主线程一旦结束，守护线程也结束
recv_thread.start()

base = wd.World(cam_pos=[-3, 2, 1], lookat_pos=[1, 0, 0])
gm.gen_frame().attach_to(base)
robot_1 = rbt_1.Pca()
current_robot_mesh = robot_1.gen_meshmodel(toggle_tcpcs=True)
current_robot_mesh.attach_to(base)


def update_task(task):
    global prev_angles
    with data_lock:
        if latest_angles is None:
            return task.again
        angles_new = latest_angles.copy()
        delay = latest_time

    if prev_angles is None:                                             # 第一次没有上一帧，直接显示
        angles_display = angles_new
    else:
        alpha = 0.1                                                     # 线性插值，可调节平滑程度
        angles_display = (1-alpha) * prev_angles + alpha * angles_new

    update_robot_joints(robot_1, angles_display)
    prev_angles = angles_display.copy()
    return task.again


taskMgr.doMethodLater(0.005, update_task, "robot_update")
try:
    base.run()
except KeyboardInterrupt:
    print("程序退出")
    server.close()
