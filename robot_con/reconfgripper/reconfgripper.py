import time
import gripperhelper as gh
import drivers.devices.dh.dh_modbus_gripper as dh_modbus_gripper
import drivers.devices.dynamixel_sdk.sdk_wrapper as mw

class Reconfgripper():
    # def __init__(self,  com, baudrate):
    #     self.sub_gripper=dxl.DynamixelMotor(com, baudrate, toggle_group_sync_write=True)
    #     # self.m_gripper = dh.DIOAODIFOAIJF()
    def __init__(self, port = 'com3', baudrate = 57600, force = 10, vel = 10):
        port = port
        baudrate = baudrate
        initstate = 0
        # g_state = 0
        force = force
        vel = vel
        self.m_gripper = dh_modbus_gripper.dh_modbus_gripper()
        self.m_gripper.open(port, baudrate)
        self.init_gripper()
        while (initstate != 1):
            initstate = self.m_gripper.GetInitState()
            time.sleep(0.2)
        self.m_gripper.SetTargetPosition(500)
        self.mg_set_vel(vel)
        self.mg_set_force(force)

    def init_gripper(self):
        self.m_gripper.Initialization()
        print('Send grip init')

    def set_sdk_griper(self, gripper, com, peripheral_baud, real=False):
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
            self.lg_set_force()
            self.lg_set_vel()
        else:
            print("please set real gripper on")

    def lg_set_force(self):
        '''
        set the force for lft gripper
        '''
        self.gripper_r.set_dxl_current_limit(current_limit=10,dxl_id=1)


    def lg_set_vel(self, vel):
        '''
        set the max vel for lft gripper
        '''
        self.gripper_r.set_dxl_pro_vel(10, dxl_id=1)


    def rg_set_force(self):
        '''
        set the force for lft gripper
        '''
        self.gripper_r.set_dxl_current_limit(current_limit=10,dxl_id=2)


    def rg_set_vel(self, vel):
        '''
        set the max vel for lft gripper
        '''
        self.gripper_r.set_dxl_pro_vel(10, dxl_id=2)

    def mg_set_force(self, force):
        '''
        set the force for main gripper
        '''
        self.m_gripper.SetTargetForce(force)

    def mg_set_vel(self, vel):
        '''
        set the max vel for main gripper
        '''
        self.m_gripper.SetTargetSpeed(vel)


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


    def lg_jaw_to(self, jawwidth):
        '''
        left gripper jaws to "jawwidth"
        '''
        pass


    def rg_jaw_to(self, jawwidth):
        '''
        Right gripper jaws to "jawwidth"
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
            time.sleep(0.2)
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