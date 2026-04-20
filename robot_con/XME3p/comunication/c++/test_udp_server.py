import socket
import time
import math
import struct
import motion.trajectory.piecewisepoly_toppra as pwp

HOST = '192.168.0.100'
PORT = 11113

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_addr = (HOST, PORT)

dt = 0.01  # 发送间隔
count = 0

# 初始关节角
angles = [0.24810892884660088, 0.49314765906211483, 0.15601830972187128,
          1.9568132443708919, -0.16044869790600802, 0.5598721281991278, 1.3089934435968165]

try:
    for i in range(1800):
        # 每个点的最后一个关节比上一个多 0.01
        angles[6] += 0.001

        msg = struct.pack('7f', *angles)
        sock.sendto(msg, (HOST, PORT))
        print(f"[{count}] 已发送关节角:", angles)
        count += 1
        time.sleep(dt)

except KeyboardInterrupt:
    print("停止发送")

finally:
    sock.close()



