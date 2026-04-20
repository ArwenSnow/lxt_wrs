import socket
import numpy as np
import threading
import time

from pylibfranka import ControllerMode, JointPositions, Robot, Torques
from robot_con.Franka_research3.a_teleoperation.joint_impedance_controller import JointImpedanceController

# ===============线程：franka控制端接收数据===============
UDP_IP = '0.0.0.0'
UDP_PORT = 11113
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((UDP_IP, UDP_PORT))
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
print("UDP 服务器已启动，等待数据...")

# 全局变量
latest_angles = None
first_angles = None
data_lock = threading.Lock()
first_angles_received = threading.Event()


def udp_receiver():
    global latest_angles, first_angles
    while True:
        try:
            data, addr = server.recvfrom(1024)
            msg = np.frombuffer(data, dtype=np.float32, count=7)
            # print("接收到弧度：", msg) # 频率太高时建议注释掉这行，防止终端IO卡顿

            with data_lock:
                latest_angles = msg

            if first_angles is None:
                first_angles = msg.copy()
                first_angles_received.set()
                print("第一次收到的角度（rad）：", first_angles)

        except Exception as e:
            print(f"UDP 接收错误: {e}")


recv_thread = threading.Thread(target=udp_receiver, daemon=True)
recv_thread.start()


def franka_control_thread():
    # === 关键修正 1：先等待主端第一组数据到来，再碰机器人 ===
    print("等待接收主端第一组数据...")
    first_angles_received.wait()
    print("已收到主端数据，开始连接机器人并初始化力控...")

    robot = Robot("172.16.0.2")
    try:
        robot.set_collision_behavior(
            [100.0] * 7, [100.0] * 7, [100.0] * 6, [100.0] * 6
        )
        # 现在启动力控，dt 的计算就会是正常的 ~0.001s
        active_control = robot.start_torque_control()

        joint_stiffness = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]
        controller = JointImpedanceController(
            stiffness=joint_stiffness,
            damping=joint_damping,
            alpha=0.2,
            speed_factor=0.2,
            max_target_age=0.5
        )

        model = robot.load_model()

        motion_finished = False
        time_elapsed = 0.0

        # 初始读取，获取机器人当前真实物理位置
        robot_state, duration = active_control.readOnce()
        current_q = np.array(robot_state.q)

        # === 关键修正 2：初始化平滑目标，防止第一帧突变 ===
        # 不要直接跳到 latest_angles，而是从机器人的当前位置开始平滑追赶
        smooth_target_q = current_q.copy()

        # 设定限制：每次控制循环(约1ms)最多允许关节变化 0.001 弧度
        # 这保证了即使主端突然发来一个巨大的位移，机器人也会平滑跟过去，绝不报错
        MAX_STEP = 0.001

        while not motion_finished:
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()
            if dt == 0: dt = 0.001  # 防御性编程，避免除以0

            current_time = time.time()
            time_elapsed += dt

            q = np.array(robot_state.q)
            dq = np.array(robot_state.dq)

            # 获取主端最新发来的目标
            target_q_raw = None
            with data_lock:
                if latest_angles is not None:
                    target_q_raw = latest_angles.copy()

            # === 关键修正 3：执行平滑插值 ===
            if target_q_raw is not None:
                diff = target_q_raw - smooth_target_q
                dist = np.linalg.norm(diff)

                # 如果主端目标与当前平滑目标的距离大于安全步长，截断它！
                if dist > MAX_STEP:
                    smooth_target_q += (diff / dist) * MAX_STEP
                else:
                    smooth_target_q = target_q_raw

                controller.set_target(smooth_target_q, current_time)

            # 计算期望力矩
            tau = controller.update(q, dq, dt, current_time)

            # 添加科里奥利补偿
            coriolis = np.array(model.coriolis(robot_state))
            tau_desired = tau + coriolis

            # 转换为力矩命令并发送
            torque_command = Torques(tau_desired.tolist())
            active_control.writeOnce(torque_command)

            # 调试打印，如果终端卡顿建议降低打印频率 (如每 100 次打印一次)
            # print(f"dt: {dt:.4f}, 力矩: {tau_desired}")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if robot is not None:
            robot.stop()


control_thread = threading.Thread(target=franka_control_thread, daemon=True)
control_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("程序退出")
    server.close()