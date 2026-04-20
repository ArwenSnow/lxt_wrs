import socket
import numpy as np
import threading
import time
import sys

from pylibfranka import Robot, Torques
from robot_con.Franka_research3.a_teleoperation.joint_impedance_controller import JointImpedanceController

# from pynput import keyboard

# # ===================按下空格切换===============================
# # 程序退出标志
# exit_flag = False
#
# def on_press(key):
#     global exit_flag
#     if key == keyboard.Key.space:
#         print("空格键按下，准备退出程序...")
#         exit_flag = True
#         return False
#
# listener = keyboard.Listener(on_press=on_press)
# listener.start()
#    ===============线程：franka控制端接收数据===============
UDP_IP = '0.0.0.0'
UDP_PORT = 11113
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((UDP_IP, UDP_PORT))
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
print("UDP 服务器已启动，等待数据...")

# 全局变量
latest_angles = None
first_angles = None
data_lock = threading.Lock()                # 用于在udp线程里更新数据时锁住
first_angles_received = threading.Event()   # 等待第一组数据来，通知franka可以就位

last_recv_time = None


def udp_receiver():
    global latest_angles, first_angles, last_recv_time
    while True:                                                         # UDP需要循环执行
        try:
            data, addr = server.recvfrom(1024)
            msg = np.frombuffer(data, dtype=np.float32, count=7)
            # print("接收到弧度：", msg)
            with data_lock:                                             # 收数据时，别的线程不能来读
                latest_angles = msg

            if first_angles is None:
                first_angles = msg.copy()
                first_angles_received.set()
                print("第一次收到的角度（rad）：", first_angles)

        except Exception as e:
            print(f"UDP 接收错误: {e}")

recv_thread = threading.Thread(target=udp_receiver, daemon=True)       # 主线程一旦结束，守护线程也结束
recv_thread.start()

# ===================线程：Franka控制线程===============================
def franka_control_thread():
    print("等待接收主端第一组数据...")
    first_angles_received.wait()
    print("已收到主端数据，开始连接机器人并初始化力控...")

    robot = Robot("172.16.0.2")

    try:
        robot.set_collision_behavior(
            [100.0] * 7, [100.0] * 7, [100.0] * 6, [100.0] * 6
        )
        active_control = robot.start_torque_control()                   # 关节阻抗控制

        joint_stiffness = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]    # 阻抗控制器实例
        joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]
        controller = JointImpedanceController(
            stiffness=joint_stiffness,
            damping=joint_damping,
            alpha=0.2,
            speed_factor=0.2,
            max_target_age=0.5
        )

        model = robot.load_model()         # 用于科里奥利补偿
        motion_finished = False
        time_elapsed = 0.0
        while not motion_finished:
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()
            current_time = time.time()
            time_elapsed += dt

            q = np.array(robot_state.q)      # 获取当前状态
            dq = np.array(robot_state.dq)

            with data_lock:
                if latest_angles is not None:
                    controller.set_target(latest_angles.copy(), current_time)

            # 计算期望力矩（不含科里奥利）
            tau = controller.update(q, dq, dt, current_time)

            # 添加科里奥利补偿
            coriolis = np.array(model.coriolis(robot_state))
            tau_desired = tau + coriolis

            # 转换为力矩命令并发送
            torque_command = Torques(tau_desired.tolist())
            active_control.writeOnce(torque_command)
            # print(f"当前时间: {time_elapsed:.3f}, 力矩: {tau_desired}")

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
    finally:
        if robot is not None:
            print("正在停止机器人...")
            robot.stop()
        print("控制线程已退出")

control_thread = threading.Thread(target=franka_control_thread, daemon=True)
control_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("程序退出")
    server.close()

# control_thread = threading.Thread(target=franka_control_thread)
# control_thread.start()
#
# try:
#     while not exit_flag:
#         time.sleep(0.1)
# except KeyboardInterrupt:
#     exit_flag = True
#
# # 等待控制线程安全退出
# print("等待控制线程结束...")
# control_thread.join(timeout=5)
# if control_thread.is_alive():
#     print("控制线程未及时退出，强制结束")
#
# listener.stop()
# server.close()
# print("资源已清理，程序退出")
# sys.exit(0)
