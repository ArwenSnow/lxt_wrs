import socket
import time
import numpy as np
import socket
import time
import math
import struct

# 远程接收端 UDP 地址
# UDP_IP = 'b946f478.natappfree.cc'
# UDP_PORT = 32795

UDP_IP = '172.16.0.1'
UDP_PORT = 14000

# 创建 UDP 套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 角度列表
angles_list = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(100)
] + [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(1, 100)
]

# 发送数据
for angles in angles_list:
    # 转成 numpy float64 数组
    data = np.array(angles, dtype=np.float64)
    sock.sendto(data.tobytes(), (UDP_IP, UDP_PORT))
    print(f"已发送关节角: {angles}")
    time.sleep(0.1)

sock.close()
print("发送完成")


