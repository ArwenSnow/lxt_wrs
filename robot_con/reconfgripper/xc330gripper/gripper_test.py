import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xc
import robot_con.reconfgripper.maingripper.maingripper as mg
import robot_con.reconfgripper.xc330gripper.xc330gripper as gh
from time import sleep


if __name__ == '__main__':
    gripper = xc.xc330gripper()
    peripheral_baud = 57600
    com = 'COM3'
    ghw = gh.Xc330Gripper(gripper, com, peripheral_baud, real=True)
    ghw.rg_grasp_with_force(25)

