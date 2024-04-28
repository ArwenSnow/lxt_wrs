import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xc
import robot_con.xc330gripper.xc330gripper as gh

if __name__ == '__main__':

    gripper = xc.xc330gripper()
    peripheral_baud = 57600
    com = 'COM3'
    ghw = gh.xc330gripper(gripper, com, peripheral_baud, real=True)
    # ghw.lg_open()
    # ghw.lg_close()
    lg_jawwidth = .025
    rg_jawwidth = .028
    ghw.sync_jaw_to( rg_jawwidth , lg_jawwidth)
    ghw.current_stop()
    a = ghw.lg_get_jawwidth()
    b = ghw.rg_get_jawwidth()
    print(a)
    print(b)