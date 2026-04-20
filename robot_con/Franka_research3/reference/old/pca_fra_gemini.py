import time
import socket
import json
import numpy as np
import threading
import math
import argparse

# 引入你的自定义库
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.pca.pca as pca
import robot_sim.robots.Franka_research3.Franka_research3 as fr3

from pylibfranka import ControllerMode, JointPositions, Robot, Torques
from robot_con.Franka_research3.a_teleoperation.joint_impedance_controller import JointImpedanceController


def make_homo(rotmat, tvec):
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = tvec
    return homo


# =============== 全局共享变量与锁 ===============
# 1. UDP 数据锁
first_angles = None
latest_angles = None
prev_angles = None  # 用于主端滤波
latest_time = None
data_lock = threading.Lock()      # 用于在udp线程里更新数据时锁住

# 2. IK 计算结果锁
fr3_target_lock = threading.Lock()
fr3_conf_new = None  # 存储最新的 IK 解算结果
T_ready = threading.Event()  # 标志位：变换矩阵 T 是否已计算完毕
is_running = True  # 全局退出标志

# 3. 变换矩阵
T = None

# =============== 线程1：UDP 接收 ===============
UDP_IP = '0.0.0.0'
UDP_PORT = 14000
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((UDP_IP, UDP_PORT))
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
print("UDP 服务器已启动，等待数据...")


def udp_receiver():
    global latest_angles, first_angles
    while is_running:
        try:
            data, addr = server.recvfrom(1024)
            msg = np.frombuffer(data, dtype=np.float64, count=7)

            angles = np.array(msg)
            angles = np.deg2rad(angles)

            with data_lock:
                latest_angles = angles

            if first_angles is None:
                first_angles = angles.copy()
                print("第一次收到的角度（rad）：", first_angles)
        except Exception as e:
            if is_running:
                print(f"UDP 接收错误: {e}")


recv_thread = threading.Thread(target=udp_receiver, daemon=True)
recv_thread.start()

# =============== WRS 仿真环境初始化 ===============
base = wd.World(cam_pos=[-3, 2, 1], lookat_pos=[1, 0, 0])
gm.gen_frame().attach_to(base)

# 主手仿真模型
robot_1 = pca.Pca()
while first_angles is None:
    time.sleep(0.1)  # 等待第一帧数据
robot_1.fk('arm', jnt_values=first_angles)
p_1, r_1 = robot_1.get_gl_tcp('arm')
init_1 = make_homo(r_1, p_1)
current_robot_mesh = robot_1.gen_meshmodel(toggle_tcpcs=True)
current_robot_mesh.attach_to(base)

# 从手仿真模型
robot_2 = fr3.Franka_research3()
fr3_pos = np.array([.50682, .6, 0])
fr3_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi)
robot_2.fix_to(fr3_pos, fr3_rot)
current_robot_mesh_2 = None  # 将在控制线程初始化后生成

# =============== 线程2：Franka 硬件控制 ===============
first_target_event = threading.Event()  # 标志位：用于通知franka控制线程第一个目标已就绪

def franka_control_thread():
    global T, current_robot_mesh_2
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()
    robot = Robot(args.ip)

    try:
        # 设置碰撞阈值
        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        active_control = robot.start_torque_control()
        robot_state, duration = active_control.readOnce()
        initial_position = np.array(robot_state.q_d if hasattr(robot_state, "q_d") else robot_state.q)

        # 计算初始变换矩阵 T
        robot_2.fk("arm", initial_position)

        # 此时生成初始模型给主线程渲染
        # 注意：不要在这里直接 attach_to base，因为这是非主线程，可能导致 OpenGL 上下文冲突
        # 这里只计算数据，模型生成留给主线程，或者使用 WRS 的线程安全方法（如果支持）
        # 为简单起见，这里只计算 T

        p_2, r_2 = robot_2.get_gl_tcp('arm')
        init_2 = make_homo(r_2, p_2)
        T = init_2 @ np.linalg.inv(init_1)

        print("franka机器人初始位置：", initial_position)
        print("遥操作到fr3的转换关系是：", T)
        T_ready.set()  # 通知 IK 线程可以开始了

        # 阻抗控制器参数
        joint_stiffness = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]
        controller = JointImpedanceController(
            stiffness=joint_stiffness,
            damping=joint_damping,
            alpha=0.1,  # 增加平滑系数
            speed_factor=0.1,
            max_target_age=1.0
        )

        model = robot.load_model()

        # 等待第一个 IK 目标生成
        print("等待 IK 解算第一个目标...")
        first_target_event.wait()
        print("已收到第一个目标，开始控制循环")

        # 内部平滑用的当前目标位置
        smooth_target_q = initial_position.copy()

        # 限制每一步的最大关节变化量 (rad/step)
        # 假设控制频率 1kHz，最大速度 1.5 rad/s，则每步最大 0.0015
        MAX_STEP = 0.0005

        while is_running:
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()
            # 防止 dt 为 0
            if dt == 0: dt = 0.001
            current_time = time.time()

            q = np.array(robot_state.q)
            dq = np.array(robot_state.dq)

            # 1. 获取最新 IK 目标（非阻塞）
            latest_ik_target = None
            with fr3_target_lock:
                if fr3_conf_new is not None:
                    latest_ik_target = fr3_conf_new.copy()

            # 2. 关键修复：平滑插值 (Slew Rate Limiter)
            # 无论 IK 线程是否卡顿，这里都保证 smooth_target_q 平滑地向 latest_ik_target 移动
            if latest_ik_target is not None:
                diff = latest_ik_target - smooth_target_q
                dist = np.linalg.norm(diff)

                if dist > 1e-5:
                    # 如果距离太远，限制步长
                    if dist > MAX_STEP:
                        step = (diff / dist) * MAX_STEP
                        smooth_target_q += step
                    else:
                        smooth_target_q = latest_ik_target

            # 3. 将平滑后的目标传给控制器
            controller.set_target(smooth_target_q, current_time)

            # 4. 计算力矩
            tau = controller.update(q, dq, dt, current_time)
            coriolis = np.array(model.coriolis(robot_state))
            tau_desired = tau + coriolis

            active_control.writeOnce(Torques(tau_desired.tolist()))

    except Exception as e:
        print(f"控制线程异常: {e}")
    finally:
        if robot is not None:
            print("停止机器人控制...")
            robot.stop()


control_thread = threading.Thread(target=franka_control_thread, daemon=True)
control_thread.start()


# =============== 线程3 (NEW)：独立的 IK 计算线程 ===============
def ik_solver_thread():
    global fr3_conf_new, prev_angles

    print("等待 T 矩阵初始化...")
    T_ready.wait()
    print("IK 计算线程启动")

    # 本地变量用于滤波
    local_prev_angles = None

    while is_running:
        start_t = time.time()

        # 1. 获取主手角度
        current_master_angles = None
        with data_lock:
            if latest_angles is not None:
                current_master_angles = latest_angles.copy()

        if current_master_angles is None:
            time.sleep(0.01)
            continue

        # 2. 简单的输入滤波 (EMA)
        if local_prev_angles is None:
            target_angles = current_master_angles
        else:
            alpha = 0.2
            target_angles = (1 - alpha) * local_prev_angles + alpha * current_master_angles
        local_prev_angles = target_angles

        # 3. 计算从手目标 (FK -> Transform -> IK)
        try:
            # 计算主手 FK
            robot_1.fk('arm', jnt_values=target_angles)
            p_1, r_1 = robot_1.get_gl_tcp('arm')

            # 映射到从手空间
            new_2 = T @ make_homo(r_1, p_1)
            new_p, new_r = new_2[:3, 3], new_2[:3, :3]

            # 计算从手 IK
            # 注意：ik 方法比较耗时，这正是之前阻塞的原因
            conf = robot_2.ik(component_name='arm', tgt_pos=new_p, tgt_rotmat=new_r)

            if conf is not None:
                # 更新全局变量供控制线程使用
                with fr3_target_lock:
                    fr3_conf_new = conf.copy()

                # 通知控制线程数据已就绪
                if not first_target_event.is_set():
                    first_target_event.set()
            else:
                # print("IK 无解")
                pass

        except Exception as e:
            print(f"IK 线程出错: {e}")

        # 4. 频率控制
        # 不需要跑太快，100Hz 足够，给控制线程留出 CPU 资源
        elapsed = time.time() - start_t
        sleep_time = 0.01 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


ik_thread = threading.Thread(target=ik_solver_thread, daemon=True)
ik_thread.start()


# =============== 主线程：只负责渲染 ===============
def update_graphics(task):
    global current_robot_mesh, current_robot_mesh_2

    # 1. 更新主手模型 (仅视觉)
    with data_lock:
        if latest_angles is not None:
            # 这里不需要很精确，直接用最新的去画就行
            robot_1.fk('arm', jnt_values=latest_angles)
            if current_robot_mesh is not None:
                current_robot_mesh.detach()
            current_robot_mesh = robot_1.gen_meshmodel(toggle_tcpcs=True)
            current_robot_mesh.attach_to(base)

    # 2. 更新从手模型 (仅视觉)
    # 从 shared variable 读取，不需要重复算 IK
    display_conf = None
    with fr3_target_lock:
        if fr3_conf_new is not None:
            display_conf = fr3_conf_new.copy()

    if display_conf is not None:
        robot_2.fk('arm', jnt_values=display_conf)
        if current_robot_mesh_2 is not None:
            current_robot_mesh_2.detach()
        current_robot_mesh_2 = robot_2.gen_meshmodel(toggle_tcpcs=True)
        current_robot_mesh_2.attach_to(base)

    return task.again


# 注册渲染任务
taskMgr.doMethodLater(0.02, update_graphics, "update_graphics")

try:
    print("仿真及控制系统运行中...")
    base.run()
except KeyboardInterrupt:
    print("程序退出")
except Exception as e:
    print(f"主循环异常: {e}")
finally:
    is_running = False
    server.close()