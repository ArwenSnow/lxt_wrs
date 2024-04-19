import drivers.devices.dynamixel_sdk.sdk_wrapper as mw
import time
import math
import numpy as np

def move_line(path, id_group):
    for item in path:

      encoder = np.round(encoder)
      print(encoder)
      result = [int(encoder[1])]
      dxl_con.set_dxl_goal_pos_sync(tgt_pos_list=result, dxl_id_list=id_group)
      time.sleep(3)

peripheral_baud = 57600  #设置波特率为57600
com = 'COM3'  #设置通信接口为com3
dxl_con = mw.DynamixelMotor(com, baud_rate=peripheral_baud, toggle_group_sync_write=True)
dxl_con.set_dxl_op_mode(5, 1)  #设置id为1的电机操作模式为5

dxl_con.enable_dxl_torque(1)  #启用电机扭矩
print(dxl_con.get_dxl_pos(1))  #获取电机位置
print(dxl_con.get_dxl_vel(1))  #获取电机速度

dxl_con.set_dxl_goal_pos_sync(tgt_pos_list=[1400], dxl_id_list=[1])  #将id为1的电机目标位置设为~
a=dxl_con.get_dxl_pos(1)
print(a)
# dxl_con.set_dxl_goal_pos(2000,0)
# time.sleep(2)
# dxl_con.set_dxl_goal_pos(2000,1)
# time.sleep(2)
# dxl_con.set_dxl_goal_pos(2000,2)
time.sleep(4)
print('//')
# print(dxl_con.get_dxl_pos(1))
a=dxl_con.get_dxl_pos(1)
print(a)

time.sleep(1)
time.sleep(1)