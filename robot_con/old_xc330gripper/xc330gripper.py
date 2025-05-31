from time import sleep
import drivers.devices.dynamixel_sdk.sdk_wrapper as dxl


class xc330gripper(object):
    def __init__(self, gripper, com, peripheral_baud, real=True):
        self.gripper = gripper
        self.real = real
        self.com = com
        self.peripheral_baud = peripheral_baud
        self.init_real_gripper()

    def init_real_gripper(self):
        if self.real:
            self.gripper_r = dxl.DynamixelMotor(self.com, baud_rate=self.peripheral_baud,
                                                toggle_group_sync_write=True)
            id_list = [0, 1]
            control_mode = 16
            for i in id_list:
                self.gripper_r.set_dxl_op_mode(control_mode, i)
                self.gripper_r.enable_dxl_torque(i)
                self.gripper_r.get_dxl_pos(i)
                self.gripper_r.set_dxl_pro_vel(10, i)
        else:
            print("please set real gripper on")


    def lg_set_force(self):
        '''
        set the force for lft gripper
        '''
        self.gripper_r.set_dxl_current_limit(current_limit=1,dxl_id=1)


    def lg_set_vel(self):
        '''
        set the max vel for lft gripper
        '''
        self.gripper_r.set_dxl_pro_vel(30, dxl_id=1)


    def rg_set_force(self):
        '''
        set the force for lft gripper
        '''
        self.gripper_r.set_dxl_current_limit(current_limit=5,dxl_id=0)


    def rg_set_vel(self):
        '''
        set the max vel for lft gripper
        '''
        self.gripper_r.set_dxl_pro_vel(20, dxl_id=0)


    def lg_open(self):
        '''
        Open left gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=2790, dxl_id=1)
        sleep(5)


    def lg_close(self):
        '''
        Close left gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=3707, dxl_id=1)
        sleep(5)

    def get_pos(self):
        self.gripper_r.get_dxl_pos(0)
        print(self.gripper_r.get_dxl_pos(0))

    def set_presentpos(self,tgt_pos):
        self.gripper_r.set_dxl_present_pos(tgt_pos,0)

    def rg_open(self):
        '''
        Open right gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=1862, dxl_id=0)
        sleep(5)


    def rg_close(self):
        '''
        Close right gripper
        '''
        self.gripper_r.set_dxl_goal_pos(tgt_pos=2785, dxl_id=0)
        sleep(5)


    def lg_move_line(self,wide):
        encoder = int(-32750 * wide + 3707)
        print(encoder)
        return encoder

    def rg_move_line(self,wide):
        encoder = int(-32964 * wide + 2785)
        print(encoder)
        return encoder

    def lg_jaw_to(self, jawwidth):
        '''
        left gripper jaws to "jawwidth"
        '''
        pos = self.lg_move_line(jawwidth)
        self.gripper_r.set_dxl_goal_pos(tgt_pos=pos, dxl_id=1)
        sleep(0.02)


    def rg_jaw_to(self, jawwidth):
        '''
        Right gripper jaws to "jawwidth"
        '''
        pos = self.rg_move_line(jawwidth)
        self.gripper_r.set_dxl_goal_pos(tgt_pos=pos, dxl_id=0)
        sleep(0.02)

    def sync_jaw_to(self, rg_jawwidth , lg_jawwidth):
        '''
        Synchronize left gripper to jaws to "lg_jawwidth" and Right gripper jaws to "rg_jawwidth"
        '''
        lg_pos = self.lg_move_line(lg_jawwidth)
        rg_pos = self.rg_move_line(rg_jawwidth)
        self.gripper_r.set_dxl_goal_pos_sync(tgt_pos_list=[rg_pos,lg_pos], dxl_id_list=[0,1])
        sleep(0.02)


    def lg_get_jawwidth(self):
        '''
        Get current jawwidth of lft gripper
        '''
        encoder = self.gripper_r.get_dxl_pos(1)
        jawwidth = (encoder-3707)/-32750
        jawwidth = round(jawwidth,3)
        return jawwidth


    def rg_get_jawwidth(self):
        '''
        Get current jawwidth of right gripper
        '''
        encoder = self.gripper_r.get_dxl_pos(0)
        jawwidth = (encoder-2785)/-32964
        jawwidth = round(jawwidth,3)
        return jawwidth
        # return encoder


    def current_stop(self):
        last_position = self.gripper_r.get_dxl_pos(dxl_id=1)
        sleep(0.02)
        new_position = self.gripper_r.get_dxl_pos(dxl_id=1)
        while(last_position != new_position):
            last_position = new_position
            sleep(0.02)
            new_position = self.gripper_r.get_dxl_pos(dxl_id=1)
        self.gripper_r.set_dxl_goal_pos(tgt_pos=new_position, dxl_id=1)


    def disable_torque(self, id):
        self.gripper_r.disable_dxl_torque(dxl_id=id)



    def init(self):
        self.sync_jaw_to(0, 0)
        sleep(2)
        self.sync_jaw_to(.028, .028)
        sleep(3)

