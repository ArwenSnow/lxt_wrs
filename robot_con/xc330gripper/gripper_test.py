import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xc
import robot_con.xc330gripper.xc330gripper as gh

if __name__ == '__main__':

    gripper = xc.xc330gripper()
    peripheral_baud = 57600
    com = 'COM3'
    ghw = gh.xc330gripper(gripper, com, peripheral_baud, real=True)
    ghw.lg_open()
    ghw.lg_close()
    jawwidth = .01
    ghw.move_con(jawwidth)
    ghw.current_stop()
    # gripper.jaw_to(realwide)
    # gripper.gen_meshmodel().attach_to(base)

    # base.run()