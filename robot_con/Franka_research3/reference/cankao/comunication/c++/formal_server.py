import socket
import time
import math
import struct
import numpy as np
import robot_sim.robots.xme3p.xme3p as rbt
import motion.probabilistic.rrt_connect as rrtc
import motion.trajectory.piecewisepoly_toppra as pwp

HOST = '192.168.0.100'
PORT = 11111
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_addr = (HOST, PORT)

robot = rbt.XME3P()
rbt_rrtc = rrtc.RRTConnect(robot)

# 规划路径
start_angles = np.array([0.24810892884660088, 0.49314765906211483, 0.15601830972187128,
                         1.9568132443708919, -0.16044869790600802, 0.5598721281991278, 1.3089934435968165])
goal_angles = np.array([0, 0, 0, math.pi/2, 0, 0, 0])
obstacle_list = []
path = rbt_rrtc.plan(component_name="arm",
                     start_conf=start_angles,
                     goal_conf=goal_angles,
                     obstacle_list=obstacle_list,
                     ext_dist=0.02,
                     max_time=300)

# 规划轨迹
ttply = pwp.PiecewisePolyTOPPRA()
interpolated_path = ttply.interpolate_by_max_spdacc(path=path,
                                                    control_frequency=.001,
                                                    max_vels=np.array([10, 10, 10, 10, 10, 10, 10]),
                                                    max_accs=np.array([10, 10, 10, 10, 10, 10, 10]),
                                                    toggle_debug=False)

try:
    for q in interpolated_path:
        msg = struct.pack('7f', *q.tolist())  # np数组 → list → 解包 → 二进制
        sock.sendto(msg, (HOST, PORT))
        print(f"发送关节角度:", q)
        time.sleep(0.001)

except KeyboardInterrupt:
    print("停止发送")

finally:
    sock.close()



