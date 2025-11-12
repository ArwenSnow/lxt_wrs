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
    exca = ExtrinsicCalibrator()
    r = np.eye(3)
    t = np.array([0, 0, 0])
    a = exca.make_homo(r, t)
    print(a)



