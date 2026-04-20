import socket
import robot_sim.manipulators.gofa5.gofa5 as gf5
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np

# 创建3D世界
base = wd.World(cam_pos=[1, 1, 0.5], lookat_pos=[0, 0, .2])
gm.gen_frame(length=0.2).attach_to(base)

# 初始化GOFA5机器人
robot_s = gf5.GOFA5()

# 初始显示机器人（零位姿）
robot_mesh = robot_s.gen_meshmodel()
robot_mesh.attach_to(base)

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
current_robot_mesh = robot_mesh


def update_robot_joints(jnt_values):
    """根据关节角度更新机器人状态"""
    global current_robot_mesh

    try:
        print(f"更新关节角度: {[f'{x:.3f}' for x in jnt_values]}")

        # 移除旧的机器人模型
        if current_robot_mesh is not None:
            current_robot_mesh.detach()

        # 更新机器人关节角度（前向运动学）
        robot_s.fk(jnt_values=jnt_values)

        # 生成新的机器人模型
        new_mesh = robot_s.gen_meshmodel()
        new_mesh.attach_to(base)
        current_robot_mesh = new_mesh

        return True

    except Exception as e:
        print(f"更新机器人时出错: {e}")
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
                update_robot_joints(jnt_values)

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