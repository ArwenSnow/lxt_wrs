# udp_recv_robot_state.py
import socket
import json
import time

UDP_IP = "127.0.0.1"   # 建议：监听所有网卡
UDP_PORT = 13000     # 必须和 NATAPP 的“本地端口”一致（或你实际绑定端口）

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
sock.bind((UDP_IP, UDP_PORT))

print(f"UDP Receiver listening on {UDP_IP}:{UDP_PORT}")

last_time = None
count = 0

def fmt_list(a, n=6):
    return "[" + ", ".join(f"{x:.4f}" for x in a[:n]) + (" ..." if len(a) > n else "") + "]"

while True:
    try:
        data, addr = sock.recvfrom(65535)
        count += 1

        now = time.time()
        dt_ms = 0.0 if last_time is None else (now - last_time) * 1000.0
        last_time = now

        text = data.decode("utf-8", errors="replace").strip()
        msg = json.loads(text)

        joint_pos = msg.get("joint_pos")   # 7
        tcp_pose = msg.get("tcp_pose")     # 16 (4x4 matrix flattened)

        if not isinstance(joint_pos, list) or len(joint_pos) != 7:
            print(f"[{count:06d}] from {addr} dt={dt_ms:.1f}ms  [WARN] joint_pos invalid: {type(joint_pos)} len={0 if joint_pos is None else len(joint_pos)}")
            continue

        if not isinstance(tcp_pose, list):
            print(f"[{count:06d}] from {addr} dt={dt_ms:.1f}ms  [WARN] tcp_pose missing/invalid: {type(tcp_pose)}")
            continue

        # 你的C++发送的是齐次矩阵(16个数)
        if len(tcp_pose) == 16:
            # Rokae 的 tcpPose 通常是按行展开：
            # [r00,r01,r02,px, r10,r11,r12,py, r20,r21,r22,pz, 0,0,0,1]
            r00, r01, r02, px = tcp_pose[0], tcp_pose[1], tcp_pose[2], tcp_pose[3]
            r10, r11, r12, py = tcp_pose[4], tcp_pose[5], tcp_pose[6], tcp_pose[7]
            r20, r21, r22, pz = tcp_pose[8], tcp_pose[9], tcp_pose[10], tcp_pose[11]

            print(f"[{count:06d}] from {addr} dt={dt_ms:.1f}ms  pos(m)=({px:.4f},{py:.4f},{pz:.4f})")
            print(f"  joint_pos(rad)={fmt_list(joint_pos, n=7)}")
            print(f"  R=\n"
                  f"    [{r00:.4f} {r01:.4f} {r02:.4f}]\n"
                  f"    [{r10:.4f} {r11:.4f} {r12:.4f}]\n"
                  f"    [{r20:.4f} {r21:.4f} {r22:.4f}]")
        else:
            # 如果你之后改成发 6 个/7 个，也不会崩
            print(f"[{count:06d}] from {addr} dt={dt_ms:.1f}ms  [WARN] tcp_pose len={len(tcp_pose)} value(head)={tcp_pose[:8]}")
            print(f"  joint_pos(rad)={fmt_list(joint_pos, n=7)}")

    except Exception as e:
        print(f"UDP receive error: {e}")
