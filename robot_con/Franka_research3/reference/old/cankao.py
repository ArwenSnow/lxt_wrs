import threading
import time
import numpy as np

# ... (保留你的 import 和 UDP 部分) ...

# ================= 共享变量 =================
# 增加一个 IK 计算专用线程的锁
ik_data_lock = threading.Lock()
shared_ik_solution = None  # 存储最新的 IK 解算结果
is_running = True  # 全局运行标志


# ================= 改进 1: 独立的 IK 计算线程 =================
# 将 IK 从 update_task (渲染线程) 中移出，单独跑
def ik_solver_thread():
    global shared_ik_solution, prev_angles
    print("IK 解算线程启动...")

    while is_running:
        loop_start = time.time()

        # 1. 获取最新的遥操作手柄角度
        current_angles = None
        with data_lock:
            if latest_angles is not None:
                current_angles = latest_angles.copy()

        if current_angles is None:
            time.sleep(0.01)
            continue

        # 2. 执行 FK (Robot 1) 和 IK (Robot 2)
        # 注意：这里不需要 update_task 里的平滑逻辑，直接算最新的一帧，
        # 平滑逻辑交给 Franka 控制端的插值器去做，这样延迟最低。
        try:
            # 这里的逻辑直接复制你的 update_robot_joints 逻辑
            robot_1.fk('arm', jnt_values=current_angles)
            p_1, r_1 = robot_1.get_gl_tcp('arm')

            # 计算目标 T
            init_1_mat = make_homo(r_1, p_1)  # 假设 init_1 已定义
            # 注意：init_1 和 init_2 需要在主线程初始化好，或者做成全局变量

            if T is not None:  # 确保 T 已计算
                new_2 = T @ init_1_mat  # 简化的逻辑，需根据你实际 T 的定义调整
                new_p, new_r = new_2[:3, 3], new_2[:3, :3]

                # 计算 IK
                conf = robot_2.ik(component_name='arm', tgt_pos=new_p, tgt_rotmat=new_r)

                if conf is not None:
                    with ik_data_lock:
                        shared_ik_solution = conf.copy()
        except Exception as e:
            print(f"IK 计算错误: {e}")

        # 控制频率：IK 不需要 1000Hz，50-100Hz 足够，
        # 留出 CPU 时间给 Franka 控制线程
        elapsed = time.time() - loop_start
        sleep_time = 0.01 - elapsed  # 目标 100Hz
        if sleep_time > 0:
            time.sleep(sleep_time)


# 启动 IK 线程
ik_thread = threading.Thread(target=ik_solver_thread, daemon=True)
ik_thread.start()


# ================= 改进 2: Franka 控制线程 (增加平滑处理) =================
def franka_control_thread():
    # ... (初始化代码保持不变) ...
    # ...

    # 定义最大允许的关节角速度 (rad/s * dt)，防止跳变
    # 假设 dt = 0.001s, max_speed = 2.0 rad/s -> max_step = 0.002
    MAX_STEP_PER_LOOP = 0.0015

    target_q = initial_position.copy()  # 内部维护的目标位置

    while not motion_finished:
        robot_state, duration = active_control.readOnce()
        # ...

        # --- 获取最新目标 ---
        new_ik_target = None
        with ik_data_lock:
            if shared_ik_solution is not None:
                new_ik_target = shared_ik_solution.copy()

        # --- 关键修改：平滑插值 ---
        # 如果 IK 线程阻塞了 0.5秒，突然发来一个很远的目标，
        # 我们不能直接 set_target，而是要一点点移过去。
        if new_ik_target is not None:
            diff = new_ik_target - target_q
            dist = np.linalg.norm(diff)

            if dist > 0.0001:  # 有差异
                if dist > MAX_STEP_PER_LOOP:
                    # 限制步长，只走一小步
                    step = (diff / dist) * MAX_STEP_PER_LOOP
                    target_q += step
                else:
                    # 距离很近，直接赋值
                    target_q = new_ik_target

            # 发送经过平滑处理的 target_q
            controller.set_target(target_q, current_time)

        # ... (计算力矩并写入 writeOnce 保持不变) ...


# ================= 改进 3: 主线程 (只负责渲染) =================
def update_task(task):
    # 主线程现在只负责“读” shared_ik_solution 并更新图形
    # 不再进行繁重的 IK 计算

    local_conf = None
    with ik_data_lock:
        if shared_ik_solution is not None:
            local_conf = shared_ik_solution.copy()

    if local_conf is not None:
        robot_2.fk(jnt_values=local_conf)
        # 更新 mesh 位置 (伪代码)
        # robot_2.gen_meshmodel()...
        pass

    return task.again