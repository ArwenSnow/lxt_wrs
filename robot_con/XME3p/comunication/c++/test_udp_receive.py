import socket
import struct

HOST = '0.0.0.0'
PORT = 14000

# 创建 UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 绑定端口
sock.bind((HOST, PORT))

print(f"UDP 接收端已启动，监听 {HOST}:{PORT}")

count = 0

while True:
    data, addr = sock.recvfrom(1024)  # 阻塞等待数据

    if len(data) == 7 * 4:  # 7 个 float，每个 4 字节
        angles = struct.unpack('7f', data)
        print(f"[{count}] 收到来自 {addr} 的关节角:", angles)
        count += 1
    else:
        print("收到异常长度数据:", len(data))
