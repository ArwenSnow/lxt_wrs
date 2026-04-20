import socket
import numpy as np

LOCAL_IP = "0.0.0.0"
LOCAL_PORT = 14000
BUFFER_SIZE = 1024

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, LOCAL_PORT))

print(f"UDP服务器监听 {LOCAL_IP}:{LOCAL_PORT} ...")

try:
    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE)
        # 将接收到的字节转回 float64 数组
        angles = np.frombuffer(data, dtype=np.float64)
        print(f"收到关节角度：{angles}")
except KeyboardInterrupt:
    print("UDP server stopped.")
finally:
    sock.close()
