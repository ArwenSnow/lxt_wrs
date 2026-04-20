import math
import socket
import time
import visualization.panda.world as wd
import modeling.geometric_model as gm
# import robot_sim.end_effectors.handgripper.finger.finger_nails.fingernails as fn
import numpy as np
import drivers.devices.dynamixel_sdk.sdk_wrapper as mw


base = wd.World(cam_pos=[1, 1, 0.5], lookat_pos=[0, 0, .2])
gm.gen_frame(length=0.2).attach_to(base)
# finger = fn.Fingernails()
peripheral_baud = 57600
com = 'COM3'
finger_r = mw.DynamixelMotor(com, baud_rate=peripheral_baud)
for i in range(5):
    finger_r.set_dxl_op_mode(op_mode=0, dxl_id=i)
    finger_r.disable_dxl_torque(i)
client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
count = 0

# 存储当前显示的手指对象，方便后续detach
current_finger_mesh = None


def update_task(task):
    global count, current_finger_mesh

    try:
        a = finger_r.get_dxl_pos(0)
        b = finger_r.get_dxl_pos(1)
        c = finger_r.get_dxl_pos(2)
        d = finger_r.get_dxl_pos(3)
        a = (a - 2048) / 4096 * 2 * math.pi
        b = -(b - 2048) / 4096 * 2 * math.pi
        c = -(c - 2048) / 4096 * 2 * math.pi
        d = (d - 2048) / 4096 * 2 * math.pi
        data_to_send = f"[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}, {0:.2f}, {0:.2f}]"
        client_socket1.send(data_to_send.encode())
        client_socket2.send(data_to_send.encode())
        print(f"发送: {data_to_send}")
    except:
        pass

    count += 0.01

    # 更新手指
    # finger.fk(jnt_values=np.array([a, b, c]))

    # 移除旧的手指mesh（如果存在）
    if current_finger_mesh is not None:
        current_finger_mesh.detach()

    # 创建新的手指mesh并附加到场景
    # current_finger_mesh = finger.gen_meshmodel()
    # current_finger_mesh.attach_to(base)

    return task.again


try:
    client_socket1.connect(('localhost', 12345))
    client_socket2.connect(('localhost', 12346))
    print("已连接到服务器")

    # 初始显示
    # current_finger_mesh = finger.gen_meshmodel()
    # current_finger_mesh.attach_to(base)

    taskMgr.doMethodLater(0.01, update_task, "update")
    base.run()

except ConnectionRefusedError:
    print("无法连接到服务器")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    client_socket1.close()
    client_socket2.close()
    print("客户端已关闭")