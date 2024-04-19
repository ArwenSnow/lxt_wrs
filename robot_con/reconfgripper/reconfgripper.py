import drivers.devices.dynamixel_sdk.sdk_wrapper as dxl
# import drivers.devices.dh

class Reconfgripper():
    def __init__(self,  com, baudrate):

        self.sub_gripper=dxl.DynamixelMotor(com, baudrate, toggle_group_sync_write=True)
        # self.m_gripper = dh.DIOAODIFOAIJF()


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
        # self.sub_gripper.set_dxl_goal_pos()
        pass

    def lg_close(self):
        '''
        Close left gripper
        '''
        pass

    def rg_open(self):
        '''
        Open right gripper
        '''
        pass

    def rg_close(self):
        '''
        Close right gripper
        '''
        pass

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