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
# import robot_con.ag145.ag145 as agctrl
import robot_con.gofa_con.gofa_con as gofa_con


def grasp_pos_rot(pos,rot):
    newrot = np.dot(rot,rm.rotmat_from_axangle(axis=np.array([0,1,0]),angle=math.pi))
    Rotation_matrix = rm.homomat_from_posrot(pos, newrot)
    gm.gen_frame(Rotation_matrix[:3, 3], Rotation_matrix[:3, :3]).attach_to(base)
    return Rotation_matrix


def prepare(pos,rot):
    new_pos = pos + np.array([0, 0, 0.1])
    newrot = np.dot(rot, rm.rotmat_from_axangle(axis=np.array([0, 1, 0]), angle=math.pi))
    Rotation_matrix = rm.homomat_from_posrot(new_pos, newrot)
    gm.gen_frame(Rotation_matrix[:3, 3], Rotation_matrix[:3, :3]).attach_to(base)
    return Rotation_matrix


def Correction_position(angle):
    object_rot = np.dot(np.dot(np.eye(3),rm.rotmat_from_axangle(axis=np.array([0,0,1]),angle=angle)),rm.rotmat_from_axangle(axis=np.array([1,0,0]),angle=math.pi))
    return object_rot


def gen_object_motion(component_name, conf_list, obj_pos, obj_rotmat, type='absolute'):
    """
    :param conf_list:
    :param obj_pos:
    :param obj_rotmat:
    :param type: 'absolute' or 'relative'
    :return:
    author: weiwei
    date: 20210125
    """
    objpose_list = []
    if type == 'absolute':
        for _ in conf_list:
            objpose_list.append(rm.homomat_from_posrot(obj_pos, obj_rotmat))
    elif type == 'relative':
        jnt_values_bk = robot_s.get_jnt_values(component_name)
        for conf in conf_list:
            robot_s.fk(component_name, conf)
            gl_obj_pos, gl_obj_rotmat = robot_s.cvt_loc_tcp_to_gl(component_name, obj_pos, obj_rotmat)
            objpose_list.append(rm.homomat_from_posrot(gl_obj_pos, gl_obj_rotmat))
        robot_s.fk(component_name, jnt_values_bk)
    else:
        raise ValueError('Type must be absolute or relative!')
    return objpose_list



def gen_object_motion2(component_name, conf_list, obj_pos, obj_rotmat):
    """
    :param conf_list:
    :param obj_pos:
    :param obj_rotmat:
    :param type: 'absolute' or 'relative'
    :return:
    author: weiwei
    date: 20210125
    """
    objpose_list = []
    past_rbt_posrot = robot_s.get_gl_tcp(component_name)
    past_rbt_posrot = rm.homomat_from_posrot(past_rbt_posrot[0],past_rbt_posrot[1])
    past_object_posrot = rm.homomat_from_posrot(obj_pos,obj_rotmat)
    past_rbt_posrot = np.linalg.inv(past_rbt_posrot)
    for conf in conf_list:
        robot_s.fk(component_name, conf)
        new_rbt_posrot = robot_s.get_gl_tcp(component_name)
        new_rbt_posrot = rm.homomat_from_posrot(new_rbt_posrot[0], new_rbt_posrot[1])
        translate = np.dot(past_rbt_posrot,new_rbt_posrot)
        new_object_posrot = np.dot(past_object_posrot,translate)
        objpose_list.append(new_object_posrot)
    return objpose_list



def interpolate_rotation_matrices(R1, R2, num_steps, include_endpoints=True):
    """
    在两个旋转矩阵之间等距生成中间旋转矩阵

    参数:
    R1, R2: 3x3 旋转矩阵
    num_steps: 生成的中间矩阵数量
    include_endpoints: 是否包含起始和结束矩阵

    返回:
    list: 包含所有旋转矩阵的列表
    """
    # 验证输入矩阵是否为3x3旋转矩阵
    assert R1.shape == (3, 3), "R1 必须是3x3矩阵"
    assert R2.shape == (3, 3), "R2 必须是3x3矩阵"
    assert np.allclose(np.dot(R1, R1.T), np.eye(3), atol=1e-6), "R1 不是正交矩阵"
    assert np.allclose(np.dot(R2, R2.T), np.eye(3), atol=1e-6), "R2 不是正交矩阵"
    assert np.allclose(np.linalg.det(R1), 1.0, atol=1e-6), "R1 行列式不为1"
    assert np.allclose(np.linalg.det(R2), 1.0, atol=1e-6), "R2 行列式不为1"

    # 转换为四元数
    rot1 = Rotation.from_matrix(R1)
    rot2 = Rotation.from_matrix(R2)

    # 创建SLERP对象
    slerp = Slerp([0, 1], Rotation.concatenate([rot1, rot2]))

    # 生成等距的时间点
    if include_endpoints:
        times = np.linspace(0, 1, num_steps + 2)
    else:
        times = np.linspace(0, 1, num_steps + 2)[1:-1]

    # 插值得到旋转
    interp_rots = slerp(times)

    # 转换为旋转矩阵
    interp_matrices = interp_rots.as_matrix()

    return list(interp_matrices)


if __name__ == '__main__':
    start = time.time()
    this_dir, this_filename = os.path.split(__file__)
    base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    robot_s = cbt.UR7E(enable_cc=True)
    # robot_s.gen_meshmodel().attach_to(base)
    # base.run()
    start_conf = np.array([0,0,0,0,0,0])
    rrtc_planner = rrtc.RRTConnect(robot_s)

    U625 = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "U625.STL"),
            cdprimit_type="box", expand_radius=.001)
    U625.set_pos(np.array([0.3, 0.3, 0.5]))
    U625_set_rot = np.eye(3)
    U625.set_rotmat(U625_set_rot)
    U625.attach_to(base)


    U625_pos = U625.get_pos()
    U625_rot = U625.get_rotmat()
    gm.gen_frame(U625_pos,U625_rot).attach_to(base)
    grasp_list = grasp_pos_rot(U625_pos,U625_rot)
    prepare_list = prepare(U625_pos,U625_rot)
    xxx = robot_s.ik(component_name='arm',tgt_pos=prepare_list[:3, 3],tgt_rotmat=prepare_list[:3, :3])
    obstacle_list = []
    grasp_path = rrtc_planner.plan(component_name="arm",
                             start_conf=start_conf,
                             goal_conf=xxx,
                             obstacle_list = obstacle_list,
                             ext_dist=0.1,
                             max_time=300)

    change_pos_list0 = np.linspace(start = prepare_list[:3, 3] , stop = grasp_list[:3, 3] , num = 20 , endpoint=True)
    grasp_path2 = []
    for i in change_pos_list0:
        grasp_path2_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list[:3, :3])
        grasp_path2_jnt = np.array([grasp_path2_jnt])
        grasp_path2.extend(grasp_path2_jnt)


    change_pos_list0 = np.linspace(start = grasp_list[:3, 3] , stop = prepare_list[:3, 3] , num = 20 , endpoint=True)
    grasp_path2x = []
    for i in change_pos_list0:
        grasp_path2x_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list[:3, :3])
        grasp_path2x_jnt = np.array([grasp_path2x_jnt])
        grasp_path2x.extend(grasp_path2x_jnt)

    ready_pos = np.array([0.313, -0.36, 0.975])
    ready_rot = np.eye(3)
    grasp_list2 = grasp_pos_rot(ready_pos,ready_rot)
    prepare_list2 = prepare(ready_pos,ready_rot)
    xxx3 = robot_s.ik(component_name='arm',tgt_pos=prepare_list2[:3, 3],tgt_rotmat=prepare_list2[:3, :3])
    robot_s.fk(component_name='arm',jnt_values=xxx3)
    # robot_s.gen_meshmodel().attach_to(base)
    grasp_path3 = rrtc_planner.plan(component_name="arm",
                             start_conf=grasp_path2x[-1],
                             goal_conf=xxx3,
                             obstacle_list = obstacle_list,
                             ext_dist=0.1,
                             max_time=300)
    change_pos_list1 = np.linspace(start = prepare_list2[:3, 3] , stop = grasp_list2[:3, 3] , num = 20 , endpoint=True)
    grasp_path4 = []
    for i in change_pos_list1:
        grasp_path2_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list2[:3, :3])
        grasp_path2_jnt = np.array([grasp_path2_jnt])
        grasp_path4.extend(grasp_path2_jnt)



    U625xx = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "u.STL"),
            cdprimit_type="box", expand_radius=.001)
    U625xx.set_pos(np.array([0.1, 0.3, 0.5]))
    U625xx_set_rot = np.eye(3)
    U625xx.set_rotmat(U625xx_set_rot)
    U625xx.attach_to(base)


    U625xx_pos = U625xx.get_pos()
    U625xx_rot = U625xx.get_rotmat()
    gm.gen_frame(U625xx_pos,U625xx_rot).attach_to(base)
    grasp_list = grasp_pos_rot(U625xx_pos,U625xx_rot)
    prepare_list = prepare(U625xx_pos,U625xx_rot)

    grasp_pathxx = []
    change_pos_listxx = np.linspace(start = grasp_list2[:3, 3] , stop = prepare_list2[:3, 3] , num = 20 , endpoint=True)
    for i in change_pos_listxx:
        grasp_pathxx_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list2[:3, :3])
        grasp_pathxx_jnt = np.array([grasp_pathxx_jnt])
        grasp_pathxx.extend(grasp_pathxx_jnt)


    xxxx = robot_s.ik(component_name='arm',tgt_pos=prepare_list[:3, 3],tgt_rotmat=prepare_list[:3, :3])
    obstacle_list = []
    grasp_path5 = rrtc_planner.plan(component_name="arm",
                             start_conf=grasp_pathxx[-1],
                             goal_conf=xxxx,
                             obstacle_list = obstacle_list,
                             ext_dist=0.1,
                             max_time=300)

    change_pos_list0 = np.linspace(start = prepare_list[:3, 3] , stop = grasp_list[:3, 3] , num = 20 , endpoint=True)
    grasp_path6 = []
    for i in change_pos_list0:
        grasp_path6_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list[:3, :3])
        grasp_path6_jnt = np.array([grasp_path6_jnt])
        grasp_path6.extend(grasp_path6_jnt)

    change_pos_list0 = np.linspace(start = grasp_list[:3, 3] , stop = prepare_list[:3, 3] , num = 20 , endpoint=True)
    grasp_path6x = []
    for i in change_pos_list0:
        grasp_path6x_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list[:3, :3])
        grasp_path6x_jnt = np.array([grasp_path6x_jnt])
        grasp_path6x.extend(grasp_path6x_jnt)


    ready_pos = np.array([-0.09468, -0.14966, 0.8527])
    ready_rot = np.eye(3)
    grasp_list2 = grasp_pos_rot(ready_pos,ready_rot)
    prepare_list2 = prepare(ready_pos,ready_rot)
    xxx3 = robot_s.ik(component_name='arm',tgt_pos=prepare_list2[:3, 3],tgt_rotmat=prepare_list2[:3, :3])
    robot_s.fk(component_name='arm',jnt_values=xxx3)
    # robot_s.gen_meshmodel().attach_to(base)
    grasp_path7 = rrtc_planner.plan(component_name="arm",
                             start_conf=grasp_path6x[-1],
                             goal_conf=xxx3,
                             obstacle_list = obstacle_list,
                             ext_dist=0.1,
                             max_time=300)
    change_pos_list1 = np.linspace(start = prepare_list2[:3, 3] , stop = grasp_list2[:3, 3] , num = 20 , endpoint=True)
    grasp_path8 = []
    xxxxxx = np.array(grasp_path7[-1])
    print(xxxxxx)
    for i in change_pos_list1:
        grasp_path6_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list2[:3, :3],seed_jnt_values=xxxxxx)
        xxxxxx = grasp_path6_jnt
        print(xxxxxx)
        grasp_path6_jnt = np.array([grasp_path6_jnt])
        grasp_path8.extend(grasp_path6_jnt)







    U625x = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "U625.STL"),
            cdprimit_type="box", expand_radius=.001)
    U625x.set_pos(np.array([0.2, 0.3, 0.5]))
    U625x_set_rot = np.eye(3)
    U625x.set_rotmat(U625x_set_rot)
    U625x.attach_to(base)


    U625x_pos = U625x.get_pos()
    U625x_rot = U625x.get_rotmat()
    gm.gen_frame(U625x_pos,U625x_rot).attach_to(base)
    grasp_list = grasp_pos_rot(U625x_pos,U625x_rot)
    prepare_list = prepare(U625x_pos,U625x_rot)

    grasp_pathx = []
    change_pos_listx = np.linspace(start = grasp_list2[:3, 3] , stop = prepare_list2[:3, 3] , num = 20 , endpoint=True)
    for i in change_pos_listx:
        grasp_pathx_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list2[:3, :3],seed_jnt_values=xxxxxx)
        grasp_pathx_jnt = np.array([grasp_pathx_jnt])
        grasp_pathx.extend(grasp_pathx_jnt)


    xxxx = robot_s.ik(component_name='arm',tgt_pos=prepare_list[:3, 3],tgt_rotmat=prepare_list[:3, :3])
    obstacle_list = []
    grasp_path9 = rrtc_planner.plan(component_name="arm",
                             start_conf=grasp_pathx[-1],
                             goal_conf=xxxx,
                             obstacle_list = obstacle_list,
                             ext_dist=0.1,
                             max_time=300)

    change_pos_list0 = np.linspace(start = prepare_list[:3, 3] , stop = grasp_list[:3, 3] , num = 20 , endpoint=True)
    grasp_path10 = []
    for i in change_pos_list0:
        grasp_path10_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list[:3, :3])
        grasp_path10_jnt = np.array([grasp_path10_jnt])
        grasp_path10.extend(grasp_path10_jnt)

    change_pos_list0 = np.linspace(start = grasp_list[:3, 3] , stop = prepare_list[:3, 3] , num = 20 , endpoint=True)
    grasp_path10x = []
    for i in change_pos_list0:
        grasp_path10x_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list[:3, :3])
        grasp_path10x_jnt = np.array([grasp_path10x_jnt])
        grasp_path10x.extend(grasp_path10x_jnt)


    ready_pos = np.array([0.213, -0.36, 0.975])
    ready_rot = np.eye(3)
    grasp_list2 = grasp_pos_rot(ready_pos,ready_rot)
    prepare_list2 = prepare(ready_pos,ready_rot)
    xxx3 = robot_s.ik(component_name='arm',tgt_pos=prepare_list2[:3, 3],tgt_rotmat=prepare_list2[:3, :3])
    robot_s.fk(component_name='arm',jnt_values=xxx3)
    # robot_s.gen_meshmodel().attach_to(base)
    grasp_path11 = rrtc_planner.plan(component_name="arm",
                             start_conf=grasp_path10x[-1],
                             goal_conf=xxx3,
                             obstacle_list = obstacle_list,
                             ext_dist=0.1,
                             max_time=300)
    change_pos_list1 = np.linspace(start = prepare_list2[:3, 3] , stop = grasp_list2[:3, 3] , num = 20 , endpoint=True)
    grasp_path12 = []
    for i in change_pos_list1:
        grasp_path10_jnt = robot_s.ik(component_name='arm',tgt_pos=i,tgt_rotmat=grasp_list2[:3, :3])
        grasp_path10_jnt = np.array([grasp_path10_jnt])
        grasp_path12.extend(grasp_path10_jnt)


    all_grasp_path = []
    all_grasp_path.extend(grasp_path)
    all_grasp_path.extend(grasp_path2)
    xx1 = len(all_grasp_path)-1
    all_grasp_path.extend(grasp_path2x)
    all_grasp_path.extend(grasp_path3)
    all_grasp_path.extend(grasp_path4)
    xx2 = len(all_grasp_path) - 1
    all_grasp_path.extend(grasp_pathxx)
    all_grasp_path.extend(grasp_path5)
    all_grasp_path.extend(grasp_path6)
    xx3 = len(all_grasp_path) - 1
    all_grasp_path.extend(grasp_path6x)
    all_grasp_path.extend(grasp_path7)
    all_grasp_path.extend(grasp_path8)
    xx4 = len(all_grasp_path) - 1
    all_grasp_path.extend(grasp_pathx)
    all_grasp_path.extend(grasp_path9)
    all_grasp_path.extend(grasp_path10)
    xx5 = len(all_grasp_path) - 1
    all_grasp_path.extend(grasp_path10x)
    all_grasp_path.extend(grasp_path11)
    all_grasp_path.extend(grasp_path12)
    robot_mesh = robot_s.gen_meshmodel()
    current_robot_mesh = robot_mesh
    count = 1
    reversible_counter = 1
    def update_robot_joints(jnt_values):
        """根据关节角度更新机器人状态"""
        global current_robot_mesh

        try:
            # 移除旧的机器人模型
            if current_robot_mesh is not None:
                current_robot_mesh.detach()

            # 更新机器人关节角度（前向运动学）
            print(jnt_values)
            robot_s.fk(jnt_values=jnt_values)

            # 生成新的机器人模型
            new_mesh = robot_s.gen_meshmodel()
            new_mesh.attach_to(base)
            current_robot_mesh = new_mesh

            return True

        except Exception as e:
            print(f"更新机器人时出错: {e}")
            return False


    def update_task(task):
        """定时任务：检查并处理网络数据"""
        global count, reversible_counter
        jnt_values = all_grasp_path[count]
        if jnt_values is not None:
            # 更新机器人状态
            update_robot_joints(jnt_values)
            if count == xx1:
                robot_s.hold(hnd_name='hnd', objcm=U625)
            if count == xx2:
                robot_s.release(hnd_name='hnd', objcm=U625)
            if count == xx3:
                robot_s.hold(hnd_name='hnd', objcm=U625xx)
            if count == xx4:
                robot_s.release(hnd_name='hnd', objcm=U625xx)
            if count == xx5:
                robot_s.hold(hnd_name='hnd', objcm=U625x)
            if count == len(all_grasp_path) -1:
                robot_s.release(hnd_name='hnd', objcm=U625x)
        count = count + reversible_counter
        if count == len(all_grasp_path)-1 or count == -1:
             reversible_counter = reversible_counter*-1
        if count == -1:
            count = 0
        return task.again


    try:
        taskMgr.doMethodLater(0.1, update_task,"update")
        base.run()
    except KeyboardInterrupt:
        print("\n服务器被用户中断")

    # for pose in path[1:-1]:
    #     robot_s.fk("arm", pose)
    #     is_collided = robot_s.is_collided(obstacle_list=[object])
    #     print(is_collided)
    #     robot_meshmodel = robot_s.gen_meshmodel()
    #     robot_meshmodel.attach_to(base)
    # base.run()
