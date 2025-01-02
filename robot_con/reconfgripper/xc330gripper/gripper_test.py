import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf
import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xc
import robot_con.reconfgripper.maingripper.maingripper as mg
import robot_con.reconfgripper.xc330gripper.xc330gripper as gh
from time import sleep


if __name__ == '__main__':
    gripper_rf = rf.reconfgripper()
    mgw = mg.MainGripper()
    sleep(2)
    mgw.jaw_to(.03)

    # gripper_xc = xc.xc330gripper()
    # com = 'COM3'
    # peripheral_baud = 57600
    # ghw = gh.Xc330Gripper(gripper_xc, com, peripheral_baud, real=True)
    # ghw.init_lg()




