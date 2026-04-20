import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.gofa5.gofa5 as cbt
import visualization.panda.world as wd
import modeling.geometric_model as gm
import time
import motion.probabilistic.rrt_connect as rrtc
import modeling.collision_model as cm
import os
from scipy.spatial.transform import Rotation, Slerp
import robot_con.ag145.ag145 as agctrl
import robot_con.gofa_con.gofa_con as gofa_con
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
if __name__ == '__main__':
        # ag_r = agctrl.Ag145driver()
        # ag_r.close_g()
        # ag_r.open_g()
        # time.sleep(3)
        # ag_r.close_g()
        # print('//')
        # rbt_r = gofa_con.GoFaArmController()
        # a = rbt_r.get_jnt_values()
        # print(a)

        # [0.81803582  0.40997784  0.22724187  0.7702138   1.19659774 - 3.52015457]
        # [0.91926492  0.598997    0.09529498  0.67526789  1.27269409 - 3.54668357]
        # [ 1.39259821  0.35744343  0.25534167 -0.62151175  1.0564478  -2.9489083 ]
        # [ 1.31911985  0.45989426  0.2427753  -0.62412974  1.01613069 -2.94908284]
        # [1.13306775  0.26843164 - 0.08098328  0.02827433  1.37427225 - 3.10040288]
        base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
        gm.gen_frame().attach_to(base)
        robot_s = cbt.GOFA5()
        result = robot_s.is_collided()
        print(result)
        a = np.array([0.81803582,  0.40997784,  0.22724187,  0.7702138,   1.19659774, - 3.52015457])
        robot_s.fk(component_name='arm',jnt_values=a)
        robot_s.gen_meshmodel().attach_to(base)
        # robot_s.show_cdprimit()
        result = robot_s.is_collided()
        print(result)
        base.run()
