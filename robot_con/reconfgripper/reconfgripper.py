import time
import drivers.devices.dynamixel_sdk.sdk_wrapper as dxl
import robot_con.reconfgripper.gripperhelper as gh

class Reconfgripper():
    def __init__(self,  com, baudrate):

        self.sub_gripper=dxl.DynamixelMotor(com, baudrate, toggle_group_sync_write=True)
        # self.m_gripper = dh.DIOAODIFOAIJF()

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
                self.gripper_r = dxl.DynamixelMotor(self.com, baud_rate=self.peripheral_baud)
                control_mode = 5
                self.gripper_r.set_dxl_op_mode(control_mode, dxl_id=1)
                self.gripper_r.enable_dxl_torque(dxl_id=1)
                self.gripper_r.get_dxl_pos(dxl_id=1)
                self.gripper_r.set_dxl_pro_vel(30, dxl_id=1)
                self.gripper_r.set_dxl_current_limit(current_limit=10, dxl_id=1)
            else:
                print("please set real gripper on")

    def lg_set_force(self, force):
        '''
        set the force for lft gripper
        '''
        pass

    def lg_set_vel(self, vel):
        '''
        set the max vel for lft gripper
        '''
        pass

    def mg_set_force(self, force):
        '''
        set the force for main gripper
        '''

        pass

    def mg_set_vel(self, vel):
        '''
        set the max vel for main gripper
        '''
        pass

    def lg_open(self):
        '''
        Open left gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1060, dxl_id=1)
        time.sleep(5)

    def lg_close(self):
        '''
        Close left gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1976, dxl_id=1)
        time.sleep(5)

    def rg_open(self):
        '''
        Open right gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1060, dxl_id=1)
        time.sleep(5)

    def rg_close(self):
        '''
        Close right gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1976, dxl_id=1)
        time.sleep(5)

    def rg_jaw_to(self, jawwidth):
        '''
        Right gripper jaws to "jawwidth"
        '''
        pass

    def lg_jaw_to(self, jawwidth):
        '''
        left gripper jaws to "jawwidth"
        '''
        pass

    def mg_open(self):
        '''
        Main gripper open
        '''
        pass

    def mg_close(self):
        '''
        Main gripper open
        '''
        pass

    def mg_jaw_to(self, jawwidth):
        '''
        Main gripper jaws to "jawwidth"
        '''
        pass

    def lg_get_jawwidth(self):
        '''
        Get current jawwidth of lft gripper
        '''
        pass

    def rg_get_jawwidth(self):
        '''
        Get current jawwidth of right gripper
        '''
        pass

    def mg_get_jawwidth(self):
        '''
        Get current jawwidth of main gripper
        '''
        pass