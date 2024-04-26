from time import sleep
import drivers.devices.dh.dh_modbus_gripper as dh

class maingripper():
    def __init__(self, port = 'com3', baudrate = 57600, force = 100, speed = 100):
        port = port
        baudrate = baudrate
        initstate = 0
        # g_state = 0
        force = force
        speed = speed
        self.m_gripper = dh.dh_modbus_gripper()
        self.m_gripper.open(port, baudrate)
        self.init_gripper()
        while (initstate != 1):
            initstate = self.m_gripper.GetInitState()
            sleep(0.2)
        self.m_gripper.SetTargetPosition(500)
        self.set_speed(speed)
        self.set_force(force)

    def init_gripper(self):
        self.m_gripper.Initialization()
        print('Send grip init')


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

    def conv2encoder(self, jawwidth):
        a = int(jawwidth * 1000 / 0.07048)
        return a

    def mg_open(self):
        '''
        Main gripper open
        '''
        self.jaw_to(0.07048)


    def mg_close(self):
        '''
        Main gripper open
        '''
        self.jaw_to(0)


    def mg_jaw_to(self, jawwidth):
        '''
        Main gripper jaws to "jawwidth"
        '''
        self.m_gripper.SetTargetPosition(self.conv2encoder(jawwidth))
        g_state = 0
        while (g_state == 0):
            g_state = self.m_gripper.GetGripState()
            sleep(0.2)


    def mg_get_jawwidth(self):
        '''
        Get current jawwidth of main gripper
        '''
        pass