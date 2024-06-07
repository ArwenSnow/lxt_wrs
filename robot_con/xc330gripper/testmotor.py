import drivers.devices.dynamixel_sdk.sdk_wrapper as mw
import time

peripheral_baud = 57600  #设置波特率为57600
com = 'COM3'  #设置通信接口为com3
dxl_con = mw.DynamixelMotor(com, baud_rate=peripheral_baud, toggle_group_sync_write=True)
dxl_con.set_dxl_op_mode(5, 0)  #设置id为1的电机操作模式为5

dxl_con.enable_dxl_torque(1)  #启用电机扭矩
time.sleep(4)
print('//')
# print(dxl_con.get_dxl_pos(1))
a=dxl_con.get_dxl_pos(1)
print(a)

time.sleep(1)
time.sleep(1)