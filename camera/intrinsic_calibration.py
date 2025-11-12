import cv2
import numpy as np
import os

CHECKERBOARD = (11, 8)  # 棋盘格内角点数量（实际行列数-1）
SQUARE_SIZE = 20  # 棋盘格实际物理尺寸（单位：毫米）
# 定义角点检测的终止条件（最大迭代30次或者误差小于0.001）
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
SAV_DIR = r"C:\Users\11154\Documents\GitHub\lxt_wrs\camera\pic"  # 采集的棋盘格图像存储目录

# 初始化三维坐标点
# objp创建了虚拟的棋盘格3D坐标（Z坐标默认为0)，形状位(81,3)的全零浮点数，对应81个内角点
# 每行存储一个角点的坐标(x,y,z)，初始z坐标默认为0（棋盘格平面）
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, 0:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE  # 生成规则的棋盘格点坐标

# 存储三维对象点和二维图像点的列表
objpoints = []  # 三维世界坐标
imgpoints = []  # 二维图像坐标

# 第一部分：处理所有标定图像
# print(os.listdir(SAV_DIR))
images = [f for f in os.listdir(SAV_DIR) if f.lower().endswith(('.bmp', '.jpg', '.jpeg'))]
if not images:
    raise FileNotFoundError(f"{SAV_DIR}目录中没有找到图像文件！")

print(f"正在处理{len(images)}张标定图像...")
for idx, fname in enumerate(images):
    img_path = os.path.join(SAV_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法读取图像{fname},已跳过")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    # ret: 布尔值，表示是否成功检测到完整角点集（全部角点且有序排列时返回True)
    # corners:
    # - 检测到的角点坐标数组，形状为(N, 1, 2)的numpy数组，N=角点总数
    # - 坐标顺序：从左到右、从上到下排列(受patternSize行列数影响)
    # - 注意：初始检测坐标不够精确，通常需配合cv2.cornerSubPix进行亚像素优化
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # 亚像素精确化（提升标定精度）
        # winSize:搜索窗口半宽尺寸，实际窗口尺寸为2*winSize+1，推荐至(11,11)或(5,5)
        # zeroZone:死区半宽尺寸，防止矩阵奇异性，推荐值(-1,-1)
        # criteria: 迭代终止条件（包含最大迭代次数和收敛精度），推荐值30次+0.001精度
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        print(f"图像 {idx + 1}/{len(images)}: 成功检测角点")
    else:
        print(f"图像 {idx + 1}/{len(images)}: 未检测到棋盘格，已跳过")


num_valid = len(objpoints)
print(f"\n有效标定图像数量：{num_valid}")
# print(f"objpoints个数: {len(objpoints)}, objpoints: {objpoints}")

# 执行标定计算
print("正在进行相机标定计算...")
# ret: float, 重投影误差，单位像素，值越小表示标定精度越高
# cameraMatrix: np.ndarray, 优化后的内参矩阵，格式同输入
# disCoeffs: np.ndarry, 优化后的畸变系数
# rvecs: 每张图像的外参旋转向量，可通过cv2.Rodrigues转换为旋转矩阵
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None,
    flags=cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST
)

# 计算重投影误差（评估标定质量）
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print(f"\n重投影误差：{mean_error / len(objpoints):.5f} pixels (越小越好)")

# 显示和保存结果
print("\n相机内参矩阵 K:")
print(mtx)
print("\n畸变系数 (k1, k2, p1, p2, k3):")
print(dist.ravel())
# 保存结果为NumPy文件
np.savez("camera_calibration_result.npz",
         mtx=mtx,
         dist=dist,
         resolution=img.shape[:2][::-1])
print("\n标定结果已保存到 camera_calibration_result.npz")

# 显示去畸变示例
img = cv2.imread(os.path.join(SAV_DIR, os.listdir(SAV_DIR)[0]))
h, w = img.shape[:2]
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)

cv2.imshow('Original vs Undistorted', np.hstack((img, undistorted)))
cv2.waitKey(1000)  # 显示3秒
cv2.destroyAllWindows()
