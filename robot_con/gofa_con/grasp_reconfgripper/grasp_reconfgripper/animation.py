def update(robot_s,                                       # 机器人模型
           finger_1_box,                                    # 物体碰撞模型，用于动画中显示物体的位置变化
           finger_2_box,
           robot_path,                                    # 机器人的运动轨迹列表，每个元素是一个关节角度数组（conf_list）
           jawwidth_path,                                 # 夹爪的开合轨迹列表，每个元素是一个数值（jawwidth_list）
           lft_jawwidth_list,
           rgt_jawwidth_list,
           lftobj_path,                                   # 物体的运动轨迹列表，每个元素是一个4×4矩阵（objpose_list）
           rgtobj_path,
           robot_attached_list,                           # 记录机器人上一帧的3D模型，在新一帧时删除，以免重叠
           object_attached_list,                          # 记录物体上一帧的3D模型，在新一帧时删除，以免重叠
           counter,                                       # 计数器（列表 counter=[0]），跟踪当前播放到哪一帧
           task):                                         # Panda3D的任务对象，用于控制循环动画

    if counter[0] >= len(robot_path):                     # 如果到达路径终点，则循环回到起点
        counter[0] = 0
    if len(robot_attached_list) != 0:                     # 清除上一帧的机器人和物体模型
        for robot_attached in robot_attached_list:
            robot_attached.detach()
        for object_attached in object_attached_list:
            object_attached.detach()
        robot_attached_list.clear()
        object_attached_list.clear()

    # 更新机器人位置
    pose = robot_path[counter[0]]                          # 取当前帧的关节角度
    robot_s.fk("arm", pose)                                # 让机器人运动到该位置
    robot_s.hnd.mg_jaw_to(jawwidth_path[counter[0]])       # 设置dh夹爪的开合
    robot_s.hnd.lft.jaw_to(lft_jawwidth_list[counter[0]])  # 设置lft的开合
    robot_s.hnd.rgt.jaw_to(rgt_jawwidth_list[counter[0]])  # 设置rgt的开合

    robot_meshmodel = robot_s.gen_meshmodel()              # 生成机器人3D模型
    robot_meshmodel.attach_to(base)                        # 把机器人模型添加到场景
    robot_attached_list.append(robot_meshmodel)            # 记录当前帧的机器人模型

    # 更新手指1位置
    lftobj_pose = lftobj_path[counter[0]]                  # 取当前帧的物体位姿
    lftobjb_copy = finger_1_box.copy()                       # 复制物体模型
    lftobjb_copy.set_homomat(lftobj_pose)                  # 设置新的位姿
    lftobjb_copy.attach_to(base)                           # 把物体添加到场景
    object_attached_list.append(lftobjb_copy)              # 记录当前帧的物体模型

    # 更新手指2位置
    rgtobj_pose = rgtobj_path[counter[0]]
    rgtobjb_copy = finger_2_box.copy()
    rgtobjb_copy.set_homomat(rgtobj_pose)
    rgtobjb_copy.attach_to(base)
    object_attached_list.append(rgtobjb_copy)
    counter[0] += 1
    return task.again                                      # 让任务继续执行


