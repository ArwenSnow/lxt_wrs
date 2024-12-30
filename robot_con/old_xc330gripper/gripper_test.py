import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xc
import robot_con.xc330gripper.xc330gripper as gh
from time import sleep

if __name__ == '__main__':

    gripper = xc.xc330gripper()
    peripheral_baud = 57600
    com = 'COM3'
    ghw = gh.xc330gripper(gripper, com, peripheral_baud, real=True)
    ghw.get_pos()
    ghw.set_presentpos(29000)
    ghw.get_pos()
    # ghw.lg_jaw_to(.0)
    # sleep(5)
    # ghw.lg_jaw_to(.028)
    # sleep(15)


    # lg_jawwidth = 0.00
    # rg_jawwidth = 0.00
    # ghw.lg_jaw_to(lg_jawwidth)
    # ghw.rg_jaw_to(rg_jawwidth)
    # # ghw.sync_jaw_to(rg_jawwidth, lg_jawwidth)
    # ghw.current_stop()
    # sleep(5)
    # a = ghw.lg_get_jawwidth()
    # b = ghw.rg_get_jawwidth()
    # print(a)
    # print(b)