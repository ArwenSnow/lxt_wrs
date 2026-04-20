import socket
import numpy as np
import threading
import time
import json
import robot_sim.robots.pca.pca as pca
import robot_con.Franka_research3.accuracy.camera_controller as cam
import keyboard


def make_homo(rotmat, pos):
    """
    将旋转矩阵和位移组合成齐次矩阵.
    """
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = pos
    return homo


UDP_IP = '0.0.0.0'
UDP_PORT = 14000
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((UDP_IP, UDP_PORT))
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
print("UDP 服务器已启动，等待数据...")

exit_flag = False
latest_angles = None
data_lock = threading.Lock()
tcp_records = []
records_lock = threading.Lock()


def udp_receiver():
    global latest_angles
    while True:
        try:
            data, addr = server.recvfrom(1024)
            msg = np.frombuffer(data, dtype=np.float64, count=7)
            # print("接收到角度：", msg)

            angles = np.array(msg)
            angles = np.deg2rad(angles)
            # print("转为弧度：", angles)

            with data_lock:
                latest_angles = angles

        except Exception as e:
            print(f"UDP 接收错误: {e}")


recv_thread = threading.Thread(target=udp_receiver, daemon=True)
recv_thread.start()

robot_s = pca.Pca()

save_dir = r"C:\Users\11154\Documents\GitHub\lxt_wrs\robot_con\Franka_research3\accuracy\pic\pictures_4"
camera_r = cam.Camera(save_directory=save_dir)
camera_thread = threading.Thread(target=camera_r.live_capture, daemon=True)
camera_thread.start()


def calculate_tcp():
    global latest_angles
    with data_lock:
        angles = latest_angles.copy() if latest_angles is not None else None
    if angles is not None:
        robot_s.fk("arm", jnt_values=angles)
        pos, rotmat = robot_s.get_gl_tcp("arm")
        homo = make_homo(rotmat, pos)
        with records_lock:
            tcp_records.append(homo.tolist())
        print(f"已记录: 齐次矩阵 = {homo.tolist()}")
    else:
        print("error")


keyboard.add_hotkey('space', calculate_tcp)
try:
    while not exit_flag:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n正在保存记录...")
    exit_flag = True

if tcp_records:
    output_file = "records/tcp_records_4.json"
    with open(output_file, 'w') as f:
        json.dump(tcp_records, f, indent=2)
    print(f"已保存 {len(tcp_records)} 条记录到 {output_file}")
else:
    print("没有记录到任何数据")

