import cv2
import numpy as np
import glob
import modeling.collision_model as cm


def make_homo(R, tvec):
    homo = np.eye(4)
    homo[:3, :3] = R
    homo[:3, 3] = tvec
    return homo


def invert_homo(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def average_homogeneous_matrices(T_list):
    Rs = [T[:3, :3] for T in T_list]      # 分离旋转和平移
    ts = [T[:3, 3] for T in T_list]

    t_mean = np.mean(ts, axis=0)          # 平移部分直接平均

    M = np.zeros((3, 3))                  # 旋转部分用 SVD 平均
    for R in Rs:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R_mean = U @ Vt

    T_mean = np.eye(4)                    # 组合
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = t_mean
    return T_mean


# 配置
intrinsic_file = 'camera_calibration_result.npz'      # 内参文件
aruco_dict_type = cv2.aruco.DICT_4X4_50               # 字典类型
marker_length = 30                                    # marker边长(mm)
image_files = glob.glob("picture/marker/3/*.bmp")   # 用来求外参的图片

# 加载相机内参
with np.load(intrinsic_file) as data:
    K = data['mtx']
    dist = data['dist']

# 创建ArUco字典与检测器
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)    # 指定的字典类型
aruco_params = cv2.aruco.DetectorParameters()                      # 指定的检测参数（阈值、边缘过滤）

mc = []
# 读取图片信息
for fname in image_files:
    img = cv2.imread(fname)                                       # 用opencv读取照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                  # 把彩色图像转为灰度图像
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)  # 创建一个ArUco marker检测器对象
    corners, ids, _ = detector.detectMarkers(gray)                # 在灰度图中检测ArUco markers，返回每个marker的四个角点坐标、id

    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)
    R, _ = cv2.Rodrigues(rvec)  # 旋转向量 → 旋转矩阵
    tvec = tvec[0][0]
    T = make_homo(R, tvec)
    mc.append(T)

# 世界坐标系下，marker原点的位置与姿态
w_0 = np.array([475, -75, 5])
w_1 = np.array([450, -75, 5])
w_2 = np.array([500, -75, 5])
w_3 = np.array([490, -75, 5])

w_4 = np.array([475, -85, 5])
w_5 = np.array([500, -85, 5])
w_6 = np.array([450, -85, 5])

w_7 = np.array([450, -62.5, 5])
w_8 = np.array([475, -62.5, 5])
w_9 = np.array([500, -62.5, 5])

w_list = [w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9]
w_rot = np.eye(3)

# 求平均外参
T_list = []
for i, value in enumerate(w_list):
    if i > 10:
        break
    T = mc[i].dot(invert_homo(make_homo(w_rot, value)))
    T_list.append(T)
T_last = average_homogeneous_matrices(T_list)

# 预测
c_predit = mc[-1]

w_predit = np.dot(invert_homo(T_last), c_predit)
x, y, z = w_predit[:3, 3]
predit_pos = np.array([x/1000, y/1000, z/1000])
print("估计值:", w_predit[:3, 3])


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])
    gm.gen_frame().attach_to(base)

    table = cm.CollisionModel("object/base.stl")
    table.set_pos(np.array([0, 0, 0]))
    table.set_rgba([.7, .7, .7, 1])
    table.attach_to(base)

    marker = cm.CollisionModel("object/marker.stl")
    marker.set_pos(predit_pos)
    marker.set_rotmat(w_predit[:3, :3])
    marker.set_rgba([1, 1, 0, 1])
    marker.attach_to(base)

    base.run()



