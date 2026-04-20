import cv2
import numpy as np
import glob


class Predit:
    def __init__(self, intrinsic_file='camera_calibration_result_12.npz', extrinsic_file='extrinsic_matrix.npz',
                 aruco_dict_type=cv2.aruco.DICT_4X4_50, marker_length=30, predit_images=None):
        self.intrinsic_file = intrinsic_file
        self.extrinsic_matrix = np.load(extrinsic_file)["T"]
        self.aruco_dict_type = aruco_dict_type
        self.marker_length = marker_length
        self.predit_images = predit_images

    def get_mtx_dist(self):
        """
        读取内参文件得到内参矩阵和畸变参数。
        """
        with np.load(self.intrinsic_file) as data:
            k = data['mtx']
            dist = data['dist']
        return k, dist

    def make_aruco(self):
        """
        为创建 aruco marker检测器提供两个参数
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)  # 指定的字典类型
        aruco_params = cv2.aruco.DetectorParameters()                         # 指定的检测参数（阈值、边缘过滤）
        return aruco_dict, aruco_params

    def read_pic(self):
        """
        读图，得到相机坐标系下的marker位姿
        """
        mc = []
        aruco_dict, aruco_params = self.make_aruco()
        for image in self.predit_images:
            img = cv2.imread(image)                                       # 用opencv读取照片
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                  # 把彩色图像转为灰度图像
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)  # 创建一个ArUco marker检测器对象
            corners, ids, _ = detector.detectMarkers(gray)                # 在灰度图中检测ArUco markers，返回每个marker的四个角点坐标、id

            k, dist = self.get_mtx_dist()
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, k, dist)
            r, _ = cv2.Rodrigues(rvec)                                    # 旋转向量 → 旋转矩阵
            tvec = tvec[0][0]
            t = self.make_homo(r, tvec)
            mc.append(t)
        return mc

    def predit(self, num):
        """
        将相机坐标系下的marker位姿转化为世界坐标系下的marker位姿
        """
        mc = self.read_pic()
        c_predit = mc[num]
        w_predit = np.dot(self.invert_homo(self.extrinsic_matrix), c_predit)
        x, y, z = w_predit[:3, 3]
        predit_pos = np.array([x / 1000, y / 1000, z / 1000])
        predit_rot = w_predit[:3, :3]
        print("估计位置:", np.array([(x+6), (y+18), (z-17)]))
        print("估计姿态:", w_predit[:3, :3])
        return predit_pos, predit_rot

    @staticmethod
    def make_homo(rotmat, tvec):
        """
        将旋转矩阵和位移向量组合成齐次矩阵形式。
        """
        homo = np.eye(4)
        homo[:3, :3] = rotmat
        homo[:3, 3] = tvec
        return homo

    @staticmethod
    def invert_homo(homo):
        """
        求齐次矩阵的逆矩阵。
        """
        rotmat = homo[:3, :3]
        tvec = homo[:3, 3]
        inv = np.eye(4)
        inv[:3, :3] = rotmat.T
        inv[:3, 3] = -rotmat.T @ tvec
        return inv


if __name__ == '__main__':
    ex_file = 'extrinsic_matrix.npz'
    image_files = glob.glob("picture/marker/7/*.bmp")

    exca = Predit(extrinsic_file=ex_file, predit_images=image_files)
    exca.predit(1)




