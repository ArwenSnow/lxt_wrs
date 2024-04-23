import robot_sim.end_effectors.gripper.reconfgripper.reconfgripper as gr
import robot_con.reconfgripper.gripperhelper as gh
import robot_con.reconfgripper.reconfgripper as rg
import drivers.devices.dynamixel_sdk.sdk_wrapper as mw

if __name__ == '__main__':

    # base = wd.World(cam_pos=[1, 1, 0.5], lookat_pos=[0, 0, .2])
    # gm.gen_frame().attach_to(base)
    gripper = gr.reconfgripper()
    # gripper.gen_meshmodel().attach_to(base)
    peripheral_baud = 57600
    com = 'COM3'
    ghw = rg.Gripperhelper(gripper, com, peripheral_baud, real=True)
    ghw.lg_open()
    ghw.lg_close()
    realwide = 0
    ghx = gh.Gripperhelper(gripper, com, peripheral_baud, real=True)
    ghx.move_con(realwide)
    ghx.current_stop()
    # gripper.jaw_to(realwide)
    # gripper.gen_meshmodel().attach_to(base)

    # base.run()