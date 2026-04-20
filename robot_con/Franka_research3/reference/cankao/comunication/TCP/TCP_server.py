import socket
import time
import json

# 连接服务器
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 50000))

print("已连接到服务器")

# 发送角度
angles_list = [
    [.50, .0, .0, .0, .0, .0, .0],
    [.51, .0, .0, .0, .0, .0, .0],
    [.52, .0, .0, .0, .0, .0, .0],
    [.53, .0, .0, .0, .0, .0, .0],
    [.54, .0, .0, .0, .0, .0, .0],
    [.55, .0, .0, .0, .0, .0, .0],
    [.56, .0, .0, .0, .0, .0, .0]
]

for angles in angles_list:
    # 添加时间戳
    msg_dict = {
        "time": time.time(),  # 秒级时间戳
        "angles": angles
    }
    msg = json.dumps(msg_dict)
    print("发送:", msg)
    client.send(msg.encode())
    time.sleep(0.5)

client.close()

