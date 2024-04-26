from time import sleep
import drivers.devices.dynamixel_sdk.sdk_wrapper as dxl

class xc330gripper(object):
    def __init__(self, gripper, com, peripheral_baud, real=False):
        self.gripper = gripper
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
            self.lg_set_force()
            self.lg_set_vel()
        else:
            print("please set real gripper on")


    def lg_set_force(self):
        '''
        set the force for lft gripper
        '''
        self.gripper_r.set_dxl_current_limit(current_limit=10,dxl_id=1)


    def lg_set_vel(self):
        '''
        set the max vel for lft gripper
        '''
        self.gripper_r.set_dxl_pro_vel(10, dxl_id=1)


    def rg_set_force(self):
        '''
        set the force for lft gripper
        '''
        self.gripper_r.set_dxl_current_limit(current_limit=10,dxl_id=2)


    def rg_set_vel(self):
        '''
        set the max vel for lft gripper
        '''
        self.gripper_r.set_dxl_pro_vel(10, dxl_id=2)


    def lg_open(self):
        '''
        Open left gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1060, dxl_id=1)
        sleep(5)


    def lg_close(self):
        '''
        Close left gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1976, dxl_id=1)
        sleep(5)


    def rg_open(self):
        '''
        Open right gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1060, dxl_id=2)
        sleep(5)


    def rg_close(self):
        '''
        Close right gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1976, dxl_id=2)
        sleep(5)


    def move_line(self,wide):
        encoder = int(-32714 * wide + 1976)
        print(encoder)
        return encoder


    def lg_jaw_to(self, jawwidth):
        '''
        left gripper jaws to "jawwidth"
        '''
        pos = self.move_line(jawwidth)
        self.gripper_r.set_dxl_goal_pos(tgt_pos=pos, dxl_id=1)
        sleep(0.02)


    def rg_jaw_to(self, jawwidth):
        '''
        Right gripper jaws to "jawwidth"
        '''
        pos = self.move_line(jawwidth)
        self.gripper_r.set_dxl_goal_pos(tgt_pos=pos, dxl_id=2)
        sleep(0.02)


    def conv2encoder(self, jawwidth):
        a = int(jawwidth * 1000 / 0.06)
        return a

    def mg_jaw_to(self, jawwidth):
        '''
        Main gripper jaws to "jawwidth"
        '''
        self.m_gripper.SetTargetPosition(self.conv2encoder(jawwidth))
        g_state = 0
        while (g_state == 0):
            g_state = self.m_gripper.GetGripState()
            sleep(0.2)
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




    def move_con(self, realwide):
        pos = self.move_line(realwide)
        self.gripper_r.set_dxl_goal_pos(tgt_pos=pos, dxl_id=1)
        sleep(0.02)

    def current_stop(self):
        last_position = self.gripper_r.get_dxl_pos(dxl_id=1)
        sleep(0.02)
        new_position = self.gripper_r.get_dxl_pos(dxl_id=1)
        while(last_position != new_position):
            last_position = new_position
            sleep(0.02)
            new_position = self.gripper_r.get_dxl_pos(dxl_id=1)
        self.gripper_r.set_dxl_goal_pos(tgt_pos=new_position, dxl_id=1)
