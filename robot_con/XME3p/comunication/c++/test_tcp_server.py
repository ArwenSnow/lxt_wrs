import socket
import time
import math
import struct

HOST = '127.0.0.1'
PORT = 12345

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("已连接到 C++ 接收端")

    angles_list = [
        [0.0, 0.0, 0.0, math.pi/2+0.01, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, math.pi/2+0.02, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, math.pi/2+0.03, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, math.pi/2+0.04, 0.0, 0.0, 0.0],
    ]

    for angles in angles_list:
        msg = struct.pack('7f', *angles)
        s.sendall(msg)
        print("发送:", angles)
        time.sleep(0.5)





