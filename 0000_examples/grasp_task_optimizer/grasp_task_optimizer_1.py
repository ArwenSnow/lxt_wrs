import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf
import robot_sim.robots.gofa5.gofa5 as gf5
import itertools

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

object_finger_list = {
    "screw": [1],
    "milk_box": [2, 3],
    "gift": [2, 3, 4],
    "box": [3, 4],
    "square_bottle": [2, 3],
    "strawberry": [5],
    "pyramid": [6, 7]
}
object_list = list(object_finger_list.keys())
n = len(object_list)

# (state, finger)
dp = [{} for _ in range(1 << n)]
pre = [{} for _ in range(1 << n)]

for i in range(n):
    for finger in object_finger_list[object_list[i]]:
        state = 1 << i
        dp[state][finger] = 0
        pre[state][finger] = (None, None)

for state in range(1 << n):
    for last_finger in dp[state]:
        for next_obj in range(n):
            if not (state & (1 << next_obj)):  # 如果next_obj还没抓过
                for next_finger in object_finger_list[object_list[next_obj]]:
                    next_state = state | (1 << next_obj)
                    cost = dp[state][last_finger] + (last_finger != next_finger)
                    if next_finger not in dp[next_state] or cost < dp[next_state][next_finger]:
                        dp[next_state][next_finger] = cost
                        pre[next_state][next_finger] = (state, last_finger)

final_state = (1 << n) - 1
end_finger = min(dp[final_state], key=lambda x: dp[final_state][x])
print(f"最少切换手指次数：{dp[final_state][end_finger]}")

path = []
state = final_state
finger = end_finger
while finger is not None:
    pre_state, pre_finger = pre[state][finger]
    grasp_obj = [i for i in range(n) if (state & (1 << i)) and (pre_state is None or not (pre_state & (1 << i)))]
    assert len(grasp_obj) == 1
    obj_name = object_list[grasp_obj[0]]
    path.append((obj_name, finger))
    state, finger = pre_state, pre_finger

path.reverse()
print("最优抓取手指路径：")
for obj, f in path:
    print(f"抓取 {obj} 用手指 {f}")

base.run()
