import socket
import struct
import time
import numpy as np
import pickle
import motion.trajectory.piecewisepoly_toppra as pwp

with open("path.pickle", "rb") as f:
    paths = pickle.load(f)
robotpath1_14 = paths["path1"]
robotpath2_14 = paths["path2"]

tpply = pwp.PiecewisePolyTOPPRA()
path1 = tpply.interpolate_by_max_spdacc(path=robotpath1_14[:, :7],
                                        control_frequency=.001,
                                        max_vels=np.ones(7) * 8,
                                        max_accs=np.ones(7) * 8,
                                        toggle_debug=False)

path2 = tpply.interpolate_by_max_spdacc(path=robotpath2_14[:, 7:],
                                        control_frequency=.001,
                                        max_vels=np.ones(7) * 8,
                                        max_accs=np.ones(7) * 8,
                                        toggle_debug=False)

size_1 = len(path1)
size_2 = len(path2)
print("path1长度为：", len(path1))
print("path2长度为：", len(path2))

# 发送到c++端
SEND_IP_1 = '169.254.160.100'
SEND_PORT_1 = 10000
SEND_IP_2 = '192.168.0.100'
SEND_PORT_2 = 10001

sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_addr_1 = (SEND_IP_1, SEND_PORT_1)
server_addr_2 = (SEND_IP_2, SEND_PORT_2)

msg_bytes1 = path1.astype(np.float32).tobytes()  # 发送轨迹长度
sender.sendto(struct.pack('I', size_1), server_addr_1)
sender.sendto(msg_bytes1, server_addr_1)
time.sleep(10)
msg_bytes2 = path2.astype(np.float32).tobytes()
sender.sendto(struct.pack('I', size_2), server_addr_2)
sender.sendto(msg_bytes2, server_addr_2)

# for i in path1:
#     try:
#         msg_bytes = struct.pack('7f', *i.astype(np.float32))  # 转 float32 发送
#         sender.sendto(msg_bytes, server_addr_1)
#         time.sleep(0.01)
#         print("发送关节角度到左臂:", i)
#         print("\n")
#     except Exception as e:
#         print(f"发送失败: {e}")
#
#
# for i in path2:
#     try:
#         msg_bytes = struct.pack('7f', *i.astype(np.float32))  # 转 float32 发送
#         sender.sendto(msg_bytes, server_addr_2)
#         time.sleep(0.01)
#         print("发送关节角度到右臂:", i)
#         print("\n")
#     except Exception as e:
#         print(f"发送失败: {e}")
