import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.ur7e.ur7e as cbt
import visualization.panda.world as wd
import modeling.geometric_model as gm
import time
import motion.probabilistic.rrt_connect as rrtc
import modeling.collision_model as cm
import os
from scipy.spatial.transform import Rotation, Slerp
import robot_con.ag145.ag145 as agctrl
import robot_con.gofa_con.gofa_con as gofa_con
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================ 配置参数类 ============================
@dataclass
class ObjectConfig:
    """物体配置参数"""
    name: str
    mesh_file: str
    start_pos: np.ndarray
    target_pos: np.ndarray
    rotation: np.ndarray = None

    def __post_init__(self):
        if self.rotation is None:
            self.rotation = np.eye(3)


@dataclass
class MotionConfig:
    """运动规划参数"""
    interpolation_points: int = 20
    rrt_ext_dist: float = 0.1
    rrt_max_time: float = 300
    animation_interval: float = 0.1
    prepare_height_offset: float = 0.1
    grasp_rotation_angle: float = math.pi


@dataclass
class RobotConfig:
    """机器人配置参数"""
    component_name: str = 'arm'
    hand_name: str = 'hnd'
    start_conf: np.ndarray = None

    def __post_init__(self):
        if self.start_conf is None:
            self.start_conf = np.array([0, 0, 0, 0, 0, 0])


# ============================ 全局配置 ============================
CONFIG = {
    'objects': [
        ObjectConfig(
            name='U625',
            mesh_file='U625.STL',
            start_pos=np.array([0.8, 0.0, 0.5]),
            target_pos=np.array([0.313, -0.36, 0.975])
        ),
        ObjectConfig(
            name='U',
            mesh_file='u.STL',
            start_pos=np.array([0.3, 0.4, 0.5]),
            target_pos=np.array([-0.09468, -0.14966, 0.8527])
        ),
        ObjectConfig(
            name='U625_2',
            mesh_file='U625.STL',
            start_pos=np.array([0.7, 0.0, 0.5]),
            target_pos=np.array([0.213, -0.36, 0.975])
        ),
        ObjectConfig(
            name='U_2',
            mesh_file='u.STL',
            start_pos=np.array([0.3, 0.5, 0.5]),
            target_pos=np.array([-0.09468, -0.04966, 0.8527])
        )
    ],
    'motion': MotionConfig(),
    'objectsbox': [
        ObjectConfig(
            name='box',
            mesh_file='box.STL',
            start_pos=np.array([0.8+0.165-0.09, 0.0-0.205, 0.5-0.04]),
            target_pos=np.array([0.313, -0.36, 0.975])
        ),
        ObjectConfig(
            name='box',
            mesh_file='box.STL',
            start_pos=np.array([0.3+0.165, 0.5+0.205-0.1, 0.5]),
            target_pos=np.array([0.313, -0.36, 0.975])
        )
    ],
    'robot': RobotConfig()
}


# ============================ 核心类 ============================
class RobotMotionPlanner:
    """机器人运动规划器"""

    def __init__(self, config: Dict):
        self.config = config
        self.robot_s = None
        self.rrt_planner = None
        self.base = None
        self.current_robot_mesh = None
        self.obstacle_list = []

    def initialize(self):
        """初始化机器人系统和可视化环境"""
        self.base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, 0.2])
        gm.gen_frame().attach_to(self.base)

        self.robot_s = cbt.UR7E(enable_cc=True)
        self.rrt_planner = rrtc.RRTConnect(self.robot_s)

    def load_objects(self) -> Dict[str, cm.CollisionModel]:
        """加载所有物体模型"""
        this_dir, _ = os.path.split(__file__)
        objects = {}

        for obj_config in self.config['objects']:
            obj = cm.CollisionModel(
                os.path.join(this_dir, "meshes", obj_config.mesh_file),
                cdprimit_type="box", expand_radius=0.001
            )
            obj.set_pos(obj_config.start_pos)
            obj.set_rotmat(obj_config.rotation)
            obj.attach_to(self.base)
            objects[obj_config.name] = obj

            # 显示坐标系
            gm.gen_frame(obj_config.start_pos, obj_config.rotation).attach_to(self.base)

        for obj_config in self.config['objectsbox']:
            obj = cm.CollisionModel(
                os.path.join(this_dir, "meshes", obj_config.mesh_file),
                cdprimit_type="box", expand_radius=0.001
            )
            obj.set_pos(obj_config.start_pos)
            obj.set_rotmat(obj_config.rotation)
            obj.set_rgba(rgba=np.array([0,0,0,1]))
            obj.attach_to(self.base)

        return objects

    @staticmethod
    def calculate_grasp_pose(pos: np.ndarray, rot: np.ndarray, config: MotionConfig) -> np.ndarray:
        """计算抓取位姿"""
        new_rot = np.dot(rot, rm.rotmat_from_axangle(axis=np.array([0, 1, 0]),
                                                     angle=config.grasp_rotation_angle))
        return rm.homomat_from_posrot(pos, new_rot)

    @staticmethod
    def calculate_prepare_pose(pos: np.ndarray, rot: np.ndarray, config: MotionConfig) -> np.ndarray:
        """计算准备位姿（抓取点上方）"""
        new_pos = pos + np.array([0, 0, config.prepare_height_offset])
        new_rot = np.dot(rot, rm.rotmat_from_axangle(axis=np.array([0, 1, 0]),
                                                     angle=config.grasp_rotation_angle))
        return rm.homomat_from_posrot(new_pos, new_rot)

    def plan_linear_motion(self, start_pose: np.ndarray, end_pose: np.ndarray,
                           config: MotionConfig, prepare:np.ndarray) -> List[np.ndarray]:
        """规划直线运动轨迹"""
        self.prepare_jnt = self.robot_s.get_jnt_values(component_name=self.config['robot'].component_name)
        start_pos = start_pose[:3, 3]
        end_pos = end_pose[:3, 3]
        rotmat = end_pose[:3, :3]

        # 位置插值
        positions = np.linspace(start=start_pos, stop=end_pos,
                                num=config.interpolation_points, endpoint=True)

        # 逆运动学求解
        path = []
        for pos in positions:
            jnt_values = self.robot_s.ik(component_name=self.config['robot'].component_name,
                                         tgt_pos=pos, tgt_rotmat=rotmat,seed_jnt_values=prepare)
            if jnt_values is not None:
                path.append(jnt_values)

        return path

    def plan_single_object_movement(self, obj: cm.CollisionModel,
                                    target_pos: np.ndarray,
                                    start_conf: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """规划单个物体的完整搬运轨迹"""
        motion_config = self.config['motion']
        robot_config = self.config['robot']

        # 获取物体当前位置和姿态
        obj_pos = obj.get_pos()
        obj_rot = obj.get_rotmat()

        # 计算抓取和准备位姿
        grasp_pose = self.calculate_grasp_pose(obj_pos, obj_rot, motion_config)
        prepare_pose = self.calculate_prepare_pose(obj_pos, obj_rot, motion_config)

        # 计算目标位姿
        target_grasp_pose = self.calculate_grasp_pose(target_pos, obj_rot, motion_config)
        target_prepare_pose = self.calculate_prepare_pose(target_pos, obj_rot, motion_config)

        # 1. 规划到准备位置
        prepare_conf = self.robot_s.ik(component_name=robot_config.component_name,
                                       tgt_pos=prepare_pose[:3, 3],
                                       tgt_rotmat=prepare_pose[:3, :3])

        path_to_prepare = self.rrt_planner.plan(
            component_name=robot_config.component_name,
            start_conf=start_conf,
            goal_conf=prepare_conf,
            obstacle_list=self.obstacle_list,
            ext_dist=motion_config.rrt_ext_dist,
            max_time=motion_config.rrt_max_time
        )
        # 2. 直线下降到抓取位置
        path_to_grasp = self.plan_linear_motion(prepare_pose, grasp_pose, motion_config,path_to_prepare[-1])

        # 3. 直线上升到准备位置
        path_to_prepare_return = self.plan_linear_motion(grasp_pose, prepare_pose, motion_config,path_to_grasp[-1])

        # 4. 规划到目标准备位置
        target_prepare_conf = self.robot_s.ik(component_name=robot_config.component_name,
                                              tgt_pos=target_prepare_pose[:3, 3],
                                              tgt_rotmat=target_prepare_pose[:3, :3])

        path_to_target_prepare = self.rrt_planner.plan(
            component_name=robot_config.component_name,
            start_conf=path_to_prepare_return[-1],
            goal_conf=target_prepare_conf,
            obstacle_list=self.obstacle_list,
            ext_dist=motion_config.rrt_ext_dist,
            max_time=motion_config.rrt_max_time
        )

        # 5. 直线下降到目标抓取位置
        path_to_target_grasp = self.plan_linear_motion(target_prepare_pose, target_grasp_pose, motion_config,path_to_target_prepare[-1])

        # 6. 直线上升到目标准备位置（释放后）
        path_to_target_prepare_return = self.plan_linear_motion(target_grasp_pose, target_prepare_pose, motion_config,path_to_target_grasp[-1])

        # 合并完整路径
        full_path = []
        full_path.extend(path_to_prepare)
        full_path.extend(path_to_grasp)
        grasp_index = len(full_path) - 1
        full_path.extend(path_to_prepare_return)
        full_path.extend(path_to_target_prepare)
        full_path.extend(path_to_target_grasp)
        release_index = len(full_path) - 1
        full_path.extend(path_to_target_prepare_return)

        return full_path, [grasp_index, release_index]

    def plan_all_movements(self, objects: Dict[str, cm.CollisionModel]) -> Tuple[
        List[np.ndarray], List[Tuple[int, str, str]]]:
        """规划所有物体的搬运轨迹"""
        full_path = []
        key_points = []  # (索引, 动作类型, 物体名)
        current_conf = self.config['robot'].start_conf

        for obj_config in self.config['objects']:
            obj = objects[obj_config.name]
            print(f"规划物体 {obj_config.name} 的搬运轨迹...")

            path, indices = self.plan_single_object_movement(
                obj, obj_config.target_pos, current_conf
            )

            # 添加关键点信息
            grasp_idx = len(full_path) + indices[0]
            release_idx = len(full_path) + indices[1]
            key_points.append((grasp_idx, 'grasp', obj_config.name))
            key_points.append((release_idx, 'release', obj_config.name))

            full_path.extend(path)
            current_conf = path[-1]  # 更新当前位置为轨迹终点

        return full_path, key_points

    def animate_movement(self, full_path: List[np.ndarray], key_points: List[Tuple[int, str, str]],
                         objects: Dict[str, cm.CollisionModel]):
        """动画演示运动轨迹"""
        current_count = 0
        reversible_counter = 1

        def update_robot_joints(jnt_values: np.ndarray) -> bool:
            """更新机器人关节状态"""
            try:
                if self.current_robot_mesh is not None:
                    self.current_robot_mesh.detach()

                self.robot_s.fk(jnt_values=jnt_values)
                new_mesh = self.robot_s.gen_meshmodel()
                new_mesh.attach_to(self.base)
                self.current_robot_mesh = new_mesh
                return True
            except Exception as e:
                print(f"更新机器人时出错: {e}")
                return False

        def execute_action(action: str, obj_name: str):
            """执行抓取或释放动作"""
            if action == 'grasp':
                self.robot_s.hold(hnd_name=self.config['robot'].hand_name,
                                  objcm=objects[obj_name],
                                  jawwidth=0.08)
                print(f"抓取物体: {obj_name}")
            elif action == 'release':
                self.robot_s.release(hnd_name=self.config['robot'].hand_name,
                                     objcm=objects[obj_name],
                                     jawwidth=0)
                print(f"释放物体: {obj_name}")

        def update_task(task):
            """定时更新任务"""
            nonlocal current_count, reversible_counter

            if current_count < len(full_path):
                jnt_values = full_path[current_count]
                update_robot_joints(jnt_values)

                # 检查并执行关键点动作
                for idx, action, obj_name in key_points:
                    if idx == current_count:
                        execute_action(action, obj_name)

                current_count += reversible_counter

                # 反转计数方向以实现循环动画
                if current_count >= len(full_path) or current_count < 0:
                    reversible_counter *= -1
                    current_count += reversible_counter * 2

            return task.again

        try:
            taskMgr.doMethodLater(self.config['motion'].animation_interval, update_task, "update")
            self.base.run()
        except KeyboardInterrupt:
            print("\n演示被用户中断")


# ============================ 主程序 ============================
def main():
    """主函数"""
    start_time = time.time()

    # 创建规划器并初始化
    planner = RobotMotionPlanner(CONFIG)
    planner.initialize()

    # 加载物体
    objects = planner.load_objects()

    # 规划所有运动轨迹
    full_path, key_points = planner.plan_all_movements(objects)

    # 显示规划时间
    planning_time = time.time() - start_time
    print(f"轨迹规划完成，耗时: {planning_time:.2f} 秒")
    print(f"总轨迹点数: {len(full_path)}")
    print(f"关键点: {key_points}")

    # 动画演示
    planner.animate_movement(full_path, key_points, objects)


if __name__ == '__main__':
    main()