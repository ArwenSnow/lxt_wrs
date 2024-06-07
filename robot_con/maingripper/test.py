import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf
import robot_con.maingripper.maingripper as mg

if __name__ == '__main__':

    gripper = rf.reconfgripper()
    peripheral_baud = 115200
    com = 'COM4'

    ghw = mg.maingripper(com, peripheral_baud)
    print('/')
    # ghw.mg_jaw_to(.04)
    ghw.jaw_to(.0)
    print('/')
    # ghw.mg_close()
    # ghw.mg_open()