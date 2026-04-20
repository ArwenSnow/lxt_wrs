import cv2
import numpy as np
import glob


class ExtrinsicCalibrator:
    def __init__(self, intrinsic_file='camera_calibration_result.npz', aruco_dict_type=cv2.aruco.DICT_4X4_50,
                 marker_length=30, image_files=None):
        self.intrinsic_file = intrinsic_file
        self.aruco_dict_type = aruco_dict_type
        self.marker_length = marker_length
        self.image_files = image_files

    def get_mtx_dist(self):
        with np.load(self.intrinsic_file) as data:
            k = data['mtx']
            dist = data['dist']
        return k, dist

    def make_aruco(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)  # 指定的字典类型
        aruco_params = cv2.aruco.DetectorParameters()                         # 指定的检测参数（阈值、边缘过滤）
        return aruco_dict, aruco_params

    def read_pic(self):
        mc = []
        aruco_dict, aruco_params = self.make_aruco()
        for image in self.image_files:
            img = cv2.imread(image)  # 用opencv读取照片
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

    def average_extrinsic_matrix(self, w_list, w_rot):
        mc = self.read_pic()
        t_list = []
        for i, value in enumerate(w_list):
            # if i > 10:
            #     break
            t = mc[i].dot(self.invert_homo(self.make_homo(w_rot, value)))
            t_list.append(t)
        t_last = self.average_homogeneous_matrices(t_list)
        return t_last

    def predit(self, t_last, num):
        mc = self.read_pic()
        # t_last = self.average_extrinsic_matrix
        c_predit = mc[num]
        w_predit = np.dot(self.invert_homo(t_last), c_predit)
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

    @staticmethod
    def average_homogeneous_matrices(homo_list):
        """
        求多个齐次矩阵的平均。
        """
        rvecs = [T[:3, :3] for T in homo_list]  # 分离旋转和平移
        tvecs = [T[:3, 3] for T in homo_list]

        t_mean = np.mean(tvecs, axis=0)         # 平移部分直接平均
        m = np.zeros((3, 3))                    # 旋转部分用SVD平均
        for R in rvecs:
            m += R
        u, _, vt = np.linalg.svd(m)
        r_mean = u @ vt

        homo_mean = np.eye(4)
        homo_mean[:3, :3] = r_mean
        homo_mean[:3, 3] = t_mean
        return homo_mean


if __name__ == '__main__':
    intrinsicfile = 'camera_calibration_result_12.npz'   # 内参文件
    aruco_type = cv2.aruco.DICT_4X4_50                # 字典类型
    length = 30                                       # marker边长(mm)
    imagefiles = glob.glob("picture/marker/6/*.bmp")  # 用来求外参的图片

    exca = ExtrinsicCalibrator(intrinsicfile, aruco_type, length, imagefiles)

    w_0 = np.array([25, 1000, 5])
    w_1 = np.array([75, 1000, 5])
    w_2 = np.array([225, 1000, 5])

    w_3 = np.array([0, 800, 5])
    w_4 = np.array([-150, 800, 5])
    w_5 = np.array([-150, 900, 5])

    w_6 = np.array([0, 1000, 5])
    w_7 = np.array([0, 1200, 5])
    w_8 = np.array([-150, 1200, 5])

    w_9 = np.array([-100, 1200, 5])
    w_10 = np.array([50, 1100, 5])
    w_11 = np.array([100, 1250, 5])

    w_12 = np.array([200, 1250, 5])
    w_13 = np.array([50, 850, 5])
    w_14 = np.array([0, 950, 5])
    w_list = [w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14]
    w_rot = np.eye(3)

    t = exca.average_extrinsic_matrix(w_list, w_rot)
    np.savez("extrinsic_matrix.npz", T=t)
    marker_pos, marker_rot = exca.predit(t, 16)









