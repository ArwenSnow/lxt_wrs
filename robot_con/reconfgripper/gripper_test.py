import os
import math
import numpy as np
import modeling.model_collection as mc
import modeling.collision_model as cm
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.reconfgripper.reconfgripper as gr
import robot_sim.end_effectors.gripper.reconfgripper.gripperhelper as gh
import robot_sim.end_effectors.gripper.gripper_interface as gp
import drivers.devices.dynamixel_sdk.sdk_wrapper as mw
import time
import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm

if __name__ == '__main__':

    # base = wd.World(cam_pos=[1, 1, 0.5], lookat_pos=[0, 0, .2])
    # gm.gen_frame().attach_to(base)
    gripper = gr.reconfgripper()
    # gripper.gen_meshmodel().attach_to(base)
    peripheral_baud = 57600
    com = 'COM3'
    ghw = gh.Gripperhelper(gripper, com, peripheral_baud, real=True)
    ghw.go_close()
    ghw.go_open()
    realwide = 0
    ghw.move_con(realwide)
    ghw.current_stop()
    # gripper.jaw_to(realwide)
    # gripper.gen_meshmodel().attach_to(base)

    # base.run()