from time import sleep
import drivers.devices.dynamixel_sdk.sdk_wrapper as dxl


class Xc330Gripper(object):
    def __init__(self, gripper, com, peripheral_baud, real=True):
        self.gripper = gripper
        self.real = real
        self.com = com
        self.peripheral_baud = peripheral_baud
        self.init_real_gripper()

    def init_real_gripper(self):
        if self.real:
            self.gripper_r = dxl.DynamixelMotor(self.com, baud_rate=self.peripheral_baud)
            id_list = [0, 1]
            control_mode = 0
            for i in id_list:
                self.gripper_r.set_dxl_op_mode(control_mode, i)
                self.gripper_r.enable_dxl_torque(i)
                self.gripper_r.set_dxl_pro_vel(20, i)
                self.gripper_r.set_dxl_current_limit(80, i)
        else:
            print("please set real gripper on")

    def init_gripper(self):
        current = int(25/61.3656*1000)
        self.gripper_r.set_dxl_goal_current(current*-1, 1, bidirection=True)
        sleep(20)
        self.gripper_r.set_dxl_goal_current(current, 1, bidirection=True)
        sleep(2)

    def lg_grasp_with_force(self, force):
        current_val = force/61.3656*1000
        current_val = int(current_val)
        print(current_val)
        self.gripper_r.set_dxl_goal_current(current_val, 1, bidirection=True)
        pre_current = self.gripper_r.get_dxl_current(1)
        print(pre_current)
        # sleep(10)

    def lg_open(self):
        default_force = 7
        current_val = default_force/61.3656*1000
        current_val = int(current_val)
        # print(current_val)
        self.gripper_r.set_dxl_goal_current(current_val, 1, bidirection=True)
        pre_current = self.gripper_r.get_dxl_current(1)
        # print(pre_current)
        sleep(1)

    def lg_close(self):
        default_force = 10
        current_val = -(default_force/61.3656*1000)
        current_val = int(current_val)
        self.gripper_r.set_dxl_goal_current(current_val, 1, bidirection=True)
        pre_current = self.gripper_r.get_dxl_current(1)
        # print(pre_current)
        sleep(1)

    def aaa(self):
        self.gripper_r.set_dxl_goal_current(-60, 0, bidirection=True)

    def rg_grasp_with_force(self, force):
        current_val = force/61.3656*1000
        current_val = int(current_val)
        # print(current_val)
        self.gripper_r.set_dxl_goal_current(current_val, 0, bidirection=True)
        pre_current = self.gripper_r.get_dxl_current(0)
        # print(pre_current)
        sleep(1)

    def rg_open(self):
        default_force = 10
        current_val = default_force / 61.3656 * 1000
        current_val = int(current_val)
        # print(current_val)
        self.gripper_r.set_dxl_goal_current(current_val, 0, bidirection=True)
        pre_current = self.gripper_r.get_dxl_current(0)
        # print(pre_current)
        sleep(1)

    def rg_close(self):
        default_force = 10
        current_val = -(default_force / 61.3656 * 1000)
        current_val = int(current_val)
        # print(current_val)
        self.gripper_r.set_dxl_goal_current(current_val, 0, bidirection=True)
        pre_current = self.gripper_r.get_dxl_current(0)
        # print(pre_current)
        sleep(1)




