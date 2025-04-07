import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xc
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf
import robot_con.reconfgripper.xc330gripper.xc330gripper as gh
import robot_con.reconfgripper.maingripper.maingripper as mg
from time import sleep


if __name__ == '__main__':
    gripper_xc = xc.xc330gripper()
    com = 'COM3'
    peripheral_baud = 57600
    ghw = gh.Xc330Gripper(gripper_xc, com, peripheral_baud, real=True)
    ghw.init_lg()
    sleep(15)
    ghw.lg_grasp_with_force(-50)
    sleep(15)
    ghw.init_rg()
    sleep(15)
    ghw.rg_grasp_with_force(-50)

    # gripper_rf = rf.reconfgripper()
    # mgw = mg.MainGripper()
    # sleep(2)
    # mgw.jaw_to(.028)






