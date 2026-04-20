import socket
import numpy as np

LOCAL_IP = "0.0.0.0"
LOCAL_PORT = 10000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, LOCAL_PORT))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
print(f"UDP 服务器已启动，监听 {LOCAL_IP}:{LOCAL_PORT} ...")

# 转发到公网
FORWARD_IP = '2ef61ee3a199fd96.natapp.cc'
FORWARD_PORT = 20257

# 打洞
try:
    sock.sendto(b'hello', (FORWARD_IP, FORWARD_PORT))
    print("已向 natapp 发送 UDP 打洞包")
except Exception as e:
    print("UDP 打洞失败:", e)

while True:
    try:
        data, addr = sock.recvfrom(1024)
        angles = np.frombuffer(data, dtype=np.float64)  # 将字节解析为 float64 数组
        print(f"收到工控机数据 {addr} -> {angles}")

        try:
            sock.sendto(data, (FORWARD_IP, FORWARD_PORT))
            print(f"已转发给 {FORWARD_IP}:{FORWARD_PORT}")
        except Exception as e:
            print(f"转发失败: {e}")

    except KeyboardInterrupt:
        print("程序退出")
        break
    except Exception as e:
        print(f"接收错误: {e}")

sock.close()
