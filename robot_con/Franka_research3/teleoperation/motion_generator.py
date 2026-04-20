import numpy as np
from typing import Tuple

class MotionGenerator:
    """运动生成器，用于生成从起始关节角到目标关节角的平滑轨迹，考虑速度/加速度限制并实现同步运动。"""

    kDeltaQMotionFinished = 1e-12   # 用于判断运动是否结束的阈值

    def __init__(self, speed_factor: float, q_start: np.ndarray, q_goal: np.ndarray):
        """
        初始化运动生成器。

        :param speed_factor: 速度因子 (0 < speed_factor ≤ 1)，用于缩放默认的最大速度和加速度。
        :param q_start: 起始关节角 (7维向量)
        :param q_goal: 目标关节角 (7维向量)
        """
        assert 0 < speed_factor <= 1, "speed_factor must be in (0, 1]"
        self.q_start = np.array(q_start, dtype=np.float64)
        self.q_goal = np.array(q_goal, dtype=np.float64)
        self.delta_q = self.q_goal - self.q_start

        # 最大速度和加速度
        self.dq_max = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]) * speed_factor
        self.ddq_max_start = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) * speed_factor
        self.ddq_max_goal = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) * speed_factor

        # 同步后的参数
        self.dq_max_sync = np.zeros(7)
        self.t_1_sync = np.zeros(7)       # 加速阶段运动时间，速度dq从0到dq_max_sync
        self.t_2_sync = np.zeros(7)       # 匀速阶段运动时间，速度dq维持在dq_max_sync
        self.t_f_sync = np.zeros(7)       # 整个阶段运动时间
        self.q_1 = np.zeros(7)            # 加速阶段位移q

        self.calculate_synchronized_values()

    def calculate_synchronized_values(self):
        """根据起始和目标差值、最大速度/加速度，计算同步后的运动参数。"""
        sign_delta_q = np.sign(self.delta_q).astype(int)         # 每个关节运动的方向
        dq_max_reach = self.dq_max.copy()
        t_f = np.zeros(7)
        delta_t_2 = np.zeros(7)
        t_1 = np.zeros(7)

        # 第一阶段：计算最大可达速度和初步所需时间
        for i in range(7):
            if abs(self.delta_q[i]) > self.kDeltaQMotionFinished:                         # 如果关节运动量大于阈值1e-12就要进行处理
                # 判断是否需要降低最大速度以满足行程
                threshold = (3.0 / 4.0 * (self.dq_max[i]**2 / self.ddq_max_start[i]) +    # 计算三角形速度曲线所需的最小距离
                             3.0 / 4.0 * (self.dq_max[i]**2 / self.ddq_max_goal[i]))
                if abs(self.delta_q[i]) < threshold:                                      # 如果这个运动量比阈值小，说明无法到达最大速度，在加速度限制下解三角形速度曲线方程计算一个新的最大速度
                    dq_max_reach[i] = np.sqrt(
                        4.0 / 3.0 * self.delta_q[i] * sign_delta_q[i] *
                        (self.ddq_max_start[i] * self.ddq_max_goal[i]) /
                        (self.ddq_max_start[i] + self.ddq_max_goal[i])
                    )
                t_1[i] = 1.5 * dq_max_reach[i] / self.ddq_max_start[i]                               # 加速阶段时间
                delta_t_2[i] = 1.5 * dq_max_reach[i] / self.ddq_max_goal[i]                          # 减速阶段时间
                t_f[i] = t_1[i] / 2.0 + delta_t_2[i] / 2.0 + abs(self.delta_q[i]) / dq_max_reach[i]  # 梯形，加速+匀速+减速，用这条梯形曲线计算出来的时间

        max_t_f = np.max(t_f)   # 选所用关节需要时间里最大的那个

        # 第二阶段：以最长时间为基准，计算同步后的参数
        for i in range(7):
            if abs(self.delta_q[i]) > self.kDeltaQMotionFinished:                         # 如果关节运动量大于阈值1e-12就要进行处理
                param_a = 1.5 / 2.0 * (self.ddq_max_goal[i] + self.ddq_max_start[i])
                param_b = -1.0 * max_t_f * self.ddq_max_goal[i] * self.ddq_max_start[i]
                param_c = abs(self.delta_q[i]) * self.ddq_max_goal[i] * self.ddq_max_start[i]
                delta = param_b * param_b - 4.0 * param_a * param_c
                if delta < 0.0:
                    delta = 0.0
                self.dq_max_sync[i] = (-1.0 * param_b - np.sqrt(delta)) / (2.0 * param_a)  # 对每个关节，根据总时间 max_t_f 和行程，求解一个二次方程，以确定同步后的最大速度 dq_max_sync[i]

                self.t_1_sync[i] = 1.5 * self.dq_max_sync[i] / self.ddq_max_start[i]      # 重新计算每个关节同步的加速时间、匀速时间和最大到达速度
                delta_t_2_sync = 1.5 * self.dq_max_sync[i] / self.ddq_max_goal[i]
                self.t_f_sync[i] = (self.t_1_sync[i] / 2.0 + delta_t_2_sync / 2.0 +
                                     abs(self.delta_q[i] / self.dq_max_sync[i]))
                self.t_2_sync[i] = self.t_f_sync[i] - delta_t_2_sync
                self.q_1[i] = self.dq_max_sync[i] * sign_delta_q[i] * (0.5 * self.t_1_sync[i])

    def calculate_desired_values(self, t: float) -> Tuple[np.ndarray, bool]:       # 输入 t 为当前时间（秒），输出一个元组：第一个是位移增量数组（7维），第二个是布尔值表示运动是否全部完成。
        """
        根据给定时间计算期望的位移增量，并返回是否完成。

        :param t: 从轨迹开始经过的时间（秒）
        :return: (delta_q_d, motion_finished)
        """
        delta_q_d = np.zeros(7)
        sign_delta_q = np.sign(self.delta_q).astype(int)    # 运动方向
        t_d = self.t_2_sync - self.t_1_sync                 # 匀速阶段持续时间 = 匀速阶段结束 - 加速阶段结束
        delta_t_2_sync = self.t_f_sync - self.t_2_sync      # 减速阶段持续时间 = 减速阶段结束 - 匀速阶段结束
        joint_motion_finished = [False] * 7                 # 计算每个关节是否完成运动

        for i in range(7):
            if abs(self.delta_q[i]) < self.kDeltaQMotionFinished:     # 若关节需要动的量的绝对值小于1e-12，则认为该关节不需要运动，位移增量设为0，并标记该关节已完成
                delta_q_d[i] = 0.0
                joint_motion_finished[i] = True
            else:
                if t < self.t_1_sync[i]:                # 加速阶段内结束
                    delta_q_d[i] = (-1.0 / (self.t_1_sync[i] ** 3) *
                                     self.dq_max_sync[i] * sign_delta_q[i] *
                                     (0.5 * t - self.t_1_sync[i]) * (t ** 3))

                elif t < self.t_2_sync[i]:              # 匀速阶段内结束
                    delta_q_d[i] = (self.q_1[i] +
                                     (t - self.t_1_sync[i]) * self.dq_max_sync[i] * sign_delta_q[i])

                elif t < self.t_f_sync[i]:              # 减速阶段内结束
                    delta_q_d[i] = (self.delta_q[i] +
                                     0.5 * (
                                         (1.0 / (delta_t_2_sync[i] ** 3) *
                                          (t - self.t_1_sync[i] - 2.0 * delta_t_2_sync[i] - t_d[i]) *
                                          ((t - self.t_1_sync[i] - t_d[i]) ** 3)) +
                                         (2.0 * t - 2.0 * self.t_1_sync[i] - delta_t_2_sync[i] - 2.0 * t_d[i])
                                     ) * self.dq_max_sync[i] * sign_delta_q[i])
                else:
                    # 结束，保持最终值
                    delta_q_d[i] = self.delta_q[i]
                    joint_motion_finished[i] = True

        motion_finished = all(joint_motion_finished)
        return delta_q_d, motion_finished

    def get_desired_joint_positions(self, trajectory_time: float) -> Tuple[np.ndarray, bool]:      # 当前轨迹持续的时间，输出一个元组：第一个是期望的关节角位置，第二个是布尔值表示所有关节均已到达目标位置。
        """
        获取当前时间对应的期望关节角。

        :param trajectory_time: 从轨迹开始经过的时间（秒）
        :return: (q_desired, motion_finished)
        """
        delta_q_d, motion_finished = self.calculate_desired_values(trajectory_time)
        q_desired = self.q_start + delta_q_d
        return q_desired, motion_finished


if __name__ == '__main__':
    speed_factor = 0.5
    q_start = np.array([1.466413121670484543e-02,-2.782122194766998291e-01,-1.398401334881782532e-02,-1.824783682823181152e+00,-1.469881180673837662e-02,1.506226897239685059e+00,2.190524898469448090e-02])
    q_end = np.array([1.466463020393814903e-02,-2.781975486172206802e-01,-1.398366668998834821e-02,-1.824798786186663246e+00,-1.469692644790069475e-02,1.506228578509641647e+00,2.190776311764839820e-02])
    motion_gen = MotionGenerator(speed_factor, q_start, q_end)
    print("可到达的最大加速度：", motion_gen.dq_max)
    print("平滑启动所需时间：", motion_gen.t_f_sync)