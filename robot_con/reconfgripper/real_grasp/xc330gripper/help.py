import numpy as np
import time
import drivers.devices.dynamixel_sdk.sdk_wrapper as mw
import threading
import math

class Gripperhelper(object):
    def __init__(self, gripper, com, peripheral_baud, real=False, sync = True):
        self.gripper = gripper
        print('gripper helper')
        self.real = real
        self.com = com
        self.peripheral_baud = peripheral_baud
        self.sync = sync
        self.init_real_finger()

        # self.finger_s = None
    def init_real_finger(self):
        if self.real:
            self.finger_r = mw.DynamixelMotor(self.com, baud_rate=self.peripheral_baud, toggle_group_sync_write=self.sync)
            id_list = [1]
            control_mode = 0
            for i in id_list:
                self.finger_r.set_dxl_op_mode(control_mode, i)
                self.finger_r.enable_dxl_torque(i)
                self.finger_r.get_dxl_pos(i)
                self.finger_r.set_dxl_pro_vel(300, i)
                self.finger_r.set_dxl_current_limit(1100,i)
                self.finger_r.set_dxl_goal_current(0,i)
        else:
            print("please set real finger on")


    def test(self):
        self.finger_r.set_dxl_goal_current(75,1)
        a = self.finger_r.get_dxl_pos(1)
        print(a)
        if a >40000:
            self.finger_r.set_dxl_goal_current(0,1)