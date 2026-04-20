import numpy as np
import time
from typing import Optional, Tuple

import sys
sys.path.insert(0, '/home/lruiqing/wrs_shu/robot_con/Franka_research3')
from motion_generator import MotionGenerator

class JointImpedanceController:
    """
    关节空间阻抗控制器，包含平滑启动（MotionGenerator）和实时跟踪模式。
    """

    def __init__(self, stiffness: list, damping: list, alpha: float = 0.2,
                 speed_factor: float = 0.2, max_target_age: float = 0.5):
        """
        初始化控制器。

        :param stiffness: 刚度系数列表，长度为 7
        :param damping: 阻尼系数列表，长度为 7
        :param alpha: 速度低通滤波系数 (0 < alpha ≤ 1)
        :param speed_factor: 运动生成器速度因子 (0 < speed_factor ≤ 1)
        :param max_target_age: 目标最大允许时间（秒），超过此时间则标记为无效
        """
        self.K = np.array(stiffness, dtype=np.float64)
        self.D = np.array(damping, dtype=np.float64)
        self.alpha = alpha
        self.speed_factor = speed_factor
        self.max_target_age = max_target_age

        # 滤波后的速度
        self.dq_filtered = np.zeros(7)

        # 目标相关
        self.latest_target: Optional[np.ndarray] = None   # 保存从外部（UDP 接收线程）接收到的最新目标关节角
        self.target_valid = False                         # 基于时间戳判断当前latest_target是否有效，若超时False，转而使用上一次的有效目标
        self.last_target_time = 0.0                       # 记录最近一次更新目标的时间戳，用于与当前时间比较以判断目标是否超时

        # 运动生成器相关
        self.motion_gen = None                            # 存储 MotionGenerator 实例，用于平滑启动
        self.motion_gen_initialized = False               # 运动生成器是否已创建（是否已收到第一个目标）
        self.move_to_start_finished = False               # 标记平滑过渡阶段是否已完成，完成后控制器将切换至实时跟踪模式，直接使用最新目标
        self.start_time = 0.0                             # 运动生成器启动的时刻
        self.time_elapsed = 0.0                           # 从运动生成器启动到当前控制周期的时间累计，用于向 motion_gen 查询期望位置

        # 用于保持上一次目标（当目标失效时）
        self.last_q_goal: Optional[np.ndarray] = None

        # 实时跟踪里，用于对传给franka的数据做插值，避免跳变
        self.dq_max = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]) * speed_factor   # 最大速度
        self.ddq_max = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) * speed_factor  # 最大加速度
        self.max_step = self.dq_max * 0.001
        self.smooth_target = None      # 当前平滑目标位置
        self.smooth_vel = np.zeros(7)  # 当前平滑目标的速度（用于加速度限制）

    def set_target(self, target: np.ndarray, timestamp: float) -> None:
        """
        设置最新目标关节角。

        :param target: 长度为7的关节角数组
        :param timestamp: 目标生成时的时间戳（秒）
        """
        self.latest_target = target.copy()
        self.last_target_time = timestamp
        self.target_valid = True

    def update(self, q: np.ndarray, dq: np.ndarray, dt: float, current_time: float) -> np.ndarray:
        """
        每个控制周期调用一次，计算期望力矩。

        :param q: 当前关节角（7维）
        :param dq: 当前关节速度（7维）
        :param dt: 自上次调用以来的时间增量（秒）
        :param current_time: 当前时间戳（秒）
        :return: 期望关节力矩数组（7维）
        """
        # 1. 速度低通滤波
        if self.dq_filtered is None:
            self.dq_filtered = dq.copy()
        else:
            self.dq_filtered = (1 - self.alpha) * self.dq_filtered + self.alpha * dq

        # 2. 检查目标是否超时
        if self.target_valid and (current_time - self.last_target_time > self.max_target_age):
            self.target_valid = False

        # 3. 初始化阶段：等待第一个有效目标
        if not self.motion_gen_initialized:
            if self.target_valid:
                # 初始化运动生成器
                self.motion_gen = MotionGenerator(self.speed_factor, q, self.latest_target)
                self.motion_gen_initialized = True
                self.move_to_start_finished = False
                self.start_time = current_time
                self.time_elapsed = 0.0
            else:
                # 尚无目标，发送零力矩（机器人可自由移动）
                return np.zeros(7)

        # 4. 平滑过渡到第一个目标
        if not self.move_to_start_finished:
            self.time_elapsed += dt
            q_goal, self.move_to_start_finished = self.motion_gen.get_desired_joint_positions(self.time_elapsed)
        else:
            # 5. 实时跟踪模式
            # if self.target_valid:
            #     q_goal = self.latest_target
            # else:
            #     # 目标失效：保持上一次有效目标（如果存在），否则保持当前位置
            #     q_goal = self.last_q_goal if self.last_q_goal is not None else q

            if self.smooth_target is None:
                self.smooth_target = q.copy()
            if self.target_valid and self.latest_target is not None:
                diff = self.latest_target - self.smooth_target
                scale_vel = 1.0
                for i in range(7):
                    if abs(diff[i]) > self.max_step[i]:
                        scale_vel = min(scale_vel, self.max_step[i] / abs(diff[i]))
                step_vel_limited = diff * scale_vel

                desired_vel = step_vel_limited / 0.001   # 期望速度
                max_vel_change = self.ddq_max * 0.001    # 速度变化限制
                vel_diff = desired_vel - self.smooth_vel
                scale_acc = 1.0

                for i in range(7):
                    if abs(vel_diff[i]) > max_vel_change[i]:
                        scale_acc = min(scale_acc, max_vel_change[i] / abs(vel_diff[i]))
                final_vel = self.smooth_vel + vel_diff * scale_acc
                self.smooth_target += final_vel * 0.001   # 更新平滑目标
                self.smooth_vel = final_vel               # 更新平滑速度
            else:
                self.smooth_vel *= 0.9  # 指数衰减
                if np.linalg.norm(self.smooth_vel) < 1e-6:
                    self.smooth_vel = np.zeros(7)


            q_goal = self.smooth_target
        # 保存本次目标供下次失效时使用
        self.last_q_goal = q_goal

        # 6. 阻抗控制律
        error = q_goal - q
        tau = self.K * error - self.D * self.dq_filtered

        return tau