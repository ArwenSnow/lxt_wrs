import cv2
import numpy as np
import time
from camera.MvImport.MvCameraControl_class import *
from camera.MvImport.CameraParams_header import *
from ctypes import *


class HKcamera:
    def __init__(self, save_dir):
        self.camera = MvCamera()
        self.dir = save_dir

    def init(self):
        # SDK初始化
        MvCamera.MV_CC_Initialize()

        # 枚举设备
        deviceList = MV_CC_DEVICE_INFO_LIST()
        n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
        MvCamera.MV_CC_EnumDevices(n_layer_type, deviceList)
        print(f"找到 {deviceList.nDeviceNum} 台设备")
        if deviceList.nDeviceNum == 0:
            exit()

        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

        self.camera.MV_CC_CreateHandle(stDeviceList)
        self.camera.MV_CC_OpenDevice()
        width = c_uint()
        height = c_uint()
        pixel_format = c_uint()
        payload_size = c_uint()
        stParam = MVCC_INTVALUE()

        # 获取 PayloadSize
        self.camera.MV_CC_GetIntValue("PayloadSize", stParam)
        payload_size.value = stParam.nCurValue

        # 获取宽度
        self.camera.MV_CC_GetIntValue("Width", stParam)
        width.value = stParam.nCurValue

        # 获取高度
        self.camera.MV_CC_GetIntValue("Height", stParam)
        height.value = stParam.nCurValue

        print(width.value, height.value)

        # 设置像素格式
        pixel_format.value = 17301505  # RGB8
        # 或者 Mono8: pixel_format.value = 17301514

        # 设置曝光时间（单位：微秒）
        exposure_time = 30000
        self.camera.MV_CC_SetFloatValue("ExposureTime", exposure_time)

        # 开始抓图
        self.camera.MV_CC_StartGrabbing()

        # 分配缓冲区
        data_buf = (c_ubyte * payload_size.value)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()

        # 运行部分
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        return payload_size, stFrameInfo

    def take_pic(self, payload_size, stFrameInfo):
        try:
            while True:
                # 每次抓图前分配新的缓冲区
                data_buf = (c_ubyte * payload_size.value)()
                ret = self.camera.MV_CC_GetOneFrameTimeout(
                    byref(data_buf),
                    payload_size.value,
                    stFrameInfo,
                    1000
                )

                if ret == 0:
                    # 转换成 numpy 数组
                    frame = np.frombuffer(data_buf, dtype=np.uint8)
                    actual_width = stFrameInfo.nWidth
                    actual_height = stFrameInfo.nHeight

                    if stFrameInfo.enPixelType == 17301505:  # RGB8
                        expected_size = actual_width * actual_height * 3
                        if len(frame) != expected_size:
                            print(f"数据大小不匹配: 期望 {expected_size}, 实际 {len(frame)}")
                            continue
                        frame = frame.reshape((actual_height, actual_width, 3))
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    elif stFrameInfo.enPixelType == 17301514:  # Mono8
                        expected_size = actual_width * actual_height
                        if len(frame) != expected_size:
                            print(f"数据大小不匹配: 期望 {expected_size}, 实际 {len(frame)}")
                            continue
                        frame = frame.reshape((actual_height, actual_width))

                    elif stFrameInfo.enPixelType == 17301513:  # Bayer 格式
                        expected_size = actual_width * actual_height
                        if len(frame) != expected_size:
                            print(f"数据大小不匹配: 期望 {expected_size}, 实际 {len(frame)}")
                            continue
                        frame = frame.reshape((actual_height, actual_width))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)

                    else:
                        print(f"不支持的像素格式: {stFrameInfo.enPixelType}")
                        break

                    # 显示图像
                    cv2.imshow("Camera", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):  # 按q退出，按s拍照
                        break
                    elif key == ord("s"):
                        filename = f"photo_{int(time.time())}.png"

                        cv2.imwrite(self.dir + filename, frame)
                        print(f"拍照成功，保存为 {filename}")

                else:
                    print(f"获取图像失败，错误码: {ret}")
                    break

        finally:
            self.camera.MV_CC_StopGrabbing()   # 停止抓图
            self.camera.MV_CC_CloseDevice()    # 关闭设备
            self.camera.MV_CC_DestroyHandle()  # 销毁句柄
            cv2.destroyAllWindows()            # 销毁窗口


if __name__ == '__main__':
    save_dir = "C:/Users/11154/Documents/GitHub/lxt_wrs/camera/picture/test/"
    cam = HKcamera(save_dir)
    a, b = cam.init()
    cam.take_pic(a, b)




