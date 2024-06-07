if __name__ == '__main__':
    import math
    import numpy as np
    import basis.robot_math as rm
    import robot_sim.robots.gofa5_robot.gofa5_robot as gf
    import motion.probabilistic.rrt_connect as rrtc
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import time

    start = time.time()
    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    robot_s = gf.GOFA5()
    robot_s.gen_meshmodel(rgba=(0, 1, 0, 1)).attach_to(base)
    start_conf = robot_s.get_jnt_values(component_name='arm')  # 指定arm，并获取arm的关节配置
    jnt_values = start_conf + np.array([0, 0.3, 0, 0.1, 0.5, 0.5])  # 关节状态=初始关节配置+关节增量
    robot_s.fk(component_name="arm", jnt_values=jnt_values)  # 通过fk计算刚刚传入的新关节状态，返回末端执行器的位置和姿态

    tgt_pos = robot_s.get_gl_tcp("arm")[0]  # 获取指定arm的末端执行器的位置
    tgt_rotmat = robot_s.get_gl_tcp("arm")[1]  # 获取指定arm的末端执行器的姿态

    goal_pos = tgt_pos + np.array([0, 0, -0.5])  # 设定目标位置
    goal_rotmat = tgt_rotmat  # 设定目标姿态
    goal_jnt_values = robot_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)  # 根据目标tcp位置和姿态逆解
    robot_s.fk(component_name="arm", jnt_values=goal_jnt_values)  # 由逆解得到的关节状态再次正解
    robot_s.gen_meshmodel(rgba=(0, 0, 1, 1)).attach_to(base)  # 生成此时机器人的几何模型（末位）

    rrtc_planner = rrtc.RRTConnect(robot_s)  # 创建了一个rrtc规划器的实例，用robot_s进行初始化
    path = rrtc_planner.plan(component_name="arm",  # 指定要规划路径的机器人部件是arm
                             start_conf=start_conf,  # 指定了起始关节配置（机器人开始搜索路径时）
                             goal_conf=goal_jnt_values,  # 指定了目标关节配置（机器人到达目标时）
                             ext_dist=0.05,  # 机器人每次扩展的步长
                             max_time=300)  # 路径规划的最长时间

    for pose in path[1:-1]:  # 遍历路径每一个点，区间是从第二个到倒数最后二个
        robot_s.fk("arm", pose)
        robot_meshmodel = robot_s.gen_meshmodel()  # 打印模型
        robot_meshmodel.attach_to(base)  # 附加模型到世界中

    end = time.time()  # 记录程序结束时间
    print('程序运行时间为: %s Seconds' % (end - start))

    base.run()
