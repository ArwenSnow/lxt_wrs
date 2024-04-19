import numpy as np
import time
import drivers.devices.dynamixel_sdk.sdk_wrapper as mw

class Gripperhelper(object):
    def __init__(self, gripper, com, peripheral_baud, real=False):
        self.gripper = gripper
        print('gripper helper')
        self.real = real
        self.com = com
        self.peripheral_baud = peripheral_baud
        self.init_real_gripper()
    def init_real_gripper(self):
        if self.real:
            self.gripper_r = mw.DynamixelMotor(self.com, baud_rate=self.peripheral_baud)
            control_mode = 5
            self.gripper_r.set_dxl_op_mode(control_mode, dxl_id=1)
            self.gripper_r.enable_dxl_torque(dxl_id=1)
            self.gripper_r.get_dxl_pos(dxl_id=1)
            self.gripper_r.set_dxl_pro_vel(30, dxl_id=1)
            self.gripper_r.set_dxl_current_limit(current_limit=10,dxl_id=1)
        else:
            print("please set real gripper on")



    def go_open(self):
        self.gripper_r.set_dxl_goal_pos(tgt_pos=700, dxl_id=1)
        time.sleep(5)

    def go_close(self):
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1722, dxl_id=1)
        time.sleep(5)

    def move_line(self,wide):
        encoder = int(-34000 * wide + 1666)
        print(encoder)
        return encoder


    def move_con(self, realwide):
        pos = self.move_line(realwide)
        self.gripper_r.set_dxl_goal_pos(tgt_pos=pos, dxl_id=1)
        time.sleep(0.02)

    def current_stop(self):
        last_position = self.gripper_r.get_dxl_pos(dxl_id=1)
        time.sleep(0.02)
        new_position = self.gripper_r.get_dxl_pos(dxl_id=1)
        while(last_position != new_position):
            last_position = new_position
            time.sleep(0.02)
            new_position = self.gripper_r.get_dxl_pos(dxl_id=1)
        self.gripper_r.set_dxl_goal_pos(tgt_pos=new_position, dxl_id=1)