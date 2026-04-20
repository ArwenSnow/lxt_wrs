import socket
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import math
import basis.robot_math as rm
import time
import os
import robot_sim.robots.pca.pca as rbt_1
import robot_sim.robots.xme3p.xme3p as rbt_2
import robot_sim.manipulators.gofa5.gofa5 as gf5


def make_homo(rotmat, tvec):
    """
    将旋转矩阵和位移向量组合成齐次矩阵形式。
    """
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = tvec
    return homo


if __name__ == '__main__':
    start = time.time()
    this_dir, this_filename = os.path.split(__file__)
    base = wd.World(cam_pos=[-3, 2, 1], lookat_pos=[1, 0, 0])
    gm.gen_frame().attach_to(base)

    # table
    table = rbt_1.Pca()
    table.base_3.gen_meshmodel(toggle_jntscs=True).attach_to(base)

    # rbt_s
    robot_1 = gf5.GOFA5()
    rbt_1_pos = np.array([-.2, .08247, 0])
    rbt_1_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi/2)
    robot_1.fix_to(rbt_1_pos, rbt_1_rot)
    rbt_1_mesh = robot_1.gen_meshmodel(toggle_tcpcs=True)
    rbt_1_mesh.attach_to(base)

    robot_2 = rbt_2.XME3P()
    # rbt_2_pos = np.array([.50682, .8, 0])
    # rbt_2_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi)
    rbt_2_pos = np.array([.50682, .3, 0])
    rbt_2_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi)
    robot_2.fix_to(rbt_2_pos, rbt_2_rot)
    rbt_2_mesh = robot_2.gen_meshmodel(toggle_tcpcs=True)
    rbt_2_mesh.attach_to(base)

    p_detla = np.array([.3, .3, -.0])

    # 创建TCP套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 绑定地址和端口
    server_socket.bind(('localhost', 12345))

    # 开始监听连接
    server_socket.listen(1)
    print("GOFA5机器人动画服务器启动，等待连接...")
    print("等待客户端发送关节角度数据，格式如: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]")

    # 接受客户端连接
    client_socket, client_address = server_socket.accept()
    print(f"接收到来自 {client_address} 的连接")

    # 设置非阻塞模式
    client_socket.setblocking(0)

    # 存储当前机器人mesh
    current_robot_mesh_1 = rbt_1_mesh
    current_robot_mesh_2 = rbt_2_mesh


    def update_robot_joints(jnt_values):
        """根据关节角度更新机器人状态"""
        global current_robot_mesh_1

        try:
            # print(f"更新关节角度: {[f'{x:.3f}' for x in jnt_values]}")
            print(" ")

            # 移除旧的机器人模型
            if current_robot_mesh_1 is not None:
                current_robot_mesh_1.detach()

            # 更新机器人关节角度（前向运动学）
            robot_1.fk(jnt_values=jnt_values)
            p, r = robot_1.get_gl_tcp()
            rz = rm.rotmat_from_axangle(np.array([1, 0, 0]), -math.pi / 2)
            r_2 = r @ rz
            p_2, r_2 = p + p_detla, r_2

            # 生成新的机器人模型
            new_mesh = robot_1.gen_meshmodel()
            new_mesh.attach_to(base)
            current_robot_mesh_1 = new_mesh

            return True, p_2, r_2

        except Exception as e:
            print(f"更新机器人时出错: {e}")
            return False

    def update_robot_joints_2(p_2, r_2):
        """根据关节角度更新机器人状态"""
        global current_robot_mesh_2

        try:
            conf = robot_2.ik(component_name='arm', tgt_pos=p_2, tgt_rotmat=r_2)
            if conf is None or (isinstance(conf, np.ndarray) and conf.size == 0):
                print("珞石机器人无法解ik")

            # 移除旧的机器人模型
            if current_robot_mesh_2 is not None:
                current_robot_mesh_2.detach()

            # 更新机器人关节角度（前向运动学）
            robot_2.fk(jnt_values=conf)

            # 生成新的机器人模型
            new_mesh = robot_2.gen_meshmodel(toggle_tcpcs=True)
            new_mesh.attach_to(base)
            current_robot_mesh_2 = new_mesh

            return True

        except Exception as e:
            print(f"更新珞石机器人时出错: {e}")
            return False


    def parse_joint_data(data_str):
        """解析关节角度数据"""
        try:
            # 移除方括号和空格
            data_str = data_str.strip().strip('[]')
            # 分割字符串并转换为浮点数
            values = [float(x.strip()) for x in data_str.split(',')]

            if len(values) == 6:
                return np.array(values)
            else:
                print(f"错误: 需要6个关节角度，收到{len(values)}个")
                return None

        except ValueError as e:
            print(f"数据格式错误: {e}")
            return None
        except Exception as e:
            print(f"解析数据时出错: {e}")
            return None


    def update_task(task):
        """定时任务：检查并处理网络数据"""
        try:
            # 尝试接收数据
            data = client_socket.recv(1024)

            if data:
                received_data = data.decode().strip()
                # print(f"收到原始数据: {received_data}")
                # 解析关节角度数据
                jnt_values = parse_joint_data(received_data)
                if jnt_values is not None:
                    # 更新机器人状态
                    _, p_2, r_2 = update_robot_joints(jnt_values)
                    update_robot_joints_2(p_2, r_2)

        except BlockingIOError:
            # 没有数据可用，继续运行
            pass
        except ConnectionResetError:
            print("客户端断开连接")
            return task.done
        except Exception as e:
            print(f"网络通信错误: {e}")

        return task.again


    def check_connection(task):
        """检查连接状态的辅助任务"""
        try:
            # 发送空数据测试连接
            client_socket.send(b'')
        except:
            print("连接已断开")
            return task.done
        return task.again


    try:
        # 启动数据更新任务，每0.03秒检查一次数据（约33Hz）
        taskMgr.doMethodLater(0.01, update_task, "network_update")

        # 启动连接检查任务，每1秒检查一次
        taskMgr.doMethodLater(0.01, check_connection, "connection_check")

        print("开始运行动画...")
        print("等待关节角度数据输入...")

        base.run()

    except KeyboardInterrupt:
        print("\n服务器被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 关闭连接
        try:
            client_socket.close()
        except:
            pass
        try:
            server_socket.close()
        except:
            pass
        print("服务器已关闭")


