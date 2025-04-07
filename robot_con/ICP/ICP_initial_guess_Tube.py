"""Author: Yixuan Su
Date: 2024/11/19 10:23
File: ICP_initial_guess_and_R_xyzt.py
"""

import open3d as o3d
import numpy as np

source = o3d.io.read_point_cloud(r"source_normalized_Cloud_cropped.ply")
target = o3d.io.read_point_cloud(r"target_normalized.ply")
num1 = np.asarray(source.points).shape[0]
num2 = np.asarray(target.points).shape[0]
print(num1, num2)

source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])

print("显示源点云...")
o3d.visualization.draw_geometries([source], window_name="source Registration", width=1024, height=768)
print("显示目标点云...")
o3d.visualization.draw_geometries([target], window_name="target Registration", width=1024, height=768)

# initial_guess = np.array([
#     [0.51058156, 0.00491067, 0.8598153, 0.30657232],
#     [-0.0921574, 0.99453585, 0.04904545, -0.30394721],
#     [-0.8548763, -0.10428005, 0.50824422, 1.47714186],
#     [0, 0, 0, 1]
# ])

# initial_guess = np.array([[-0.69143706, -0.71064714,  0.12998249, 1.00587988],
#                           [-0.64461358, 0.52565226, -0.55512434,  0.91515372],
#                           [0.32617193, -0.46762202, -0.82154825, 0.44813821],
#                           [0, 0, 0, 1]
#                           ])

#
# initial_guess = np.array([[0.82649164, -0.06160907, -0.55956759, 1.24135564],
#                           [0.56273542, 0.06304022, 0.82422981, 2.50385857],
#                           [-0.01550476, -0.99610756, 0.08677182, 0.82665619],
#                           [0, 0, 0, 1]])


# initial_guess = np.array([[-0.82647682, -0.06199939, 0.55954637, 0.78003854],
#                           [-0.56274889, 0.06308269, -0.82421736, 1.22544392],
#                           [0.01580328, -0.99608065, -0.08702647, 0.77418869],
#                           [0, 0, 0, 1]])

# initial_guess = np.array([[0.830, 0.113, 0.55978426, 1.0370999],
#                           [-0.56296371, 0.06471515, - 0.82394405, 1.1727418],
#                           [0.01441426, -0.99600926, - 0.0880783, 0.83911458],
#                           [0, 0, 0, 1]])


# initial_guess = np.array([[0.830, 0.113, -0.538, -17.107],
#                           [-0.548, 0.081, -0.828, -14.283],
#                           [-0.051, 0.986, 0.130, -3.566],
#                           [0, 0, 0, 1]])


initial_guess = np.array([[0.99101628, 0.00453221, 0.13366447, -0.00685519],
                          [-0.00927114, 0.99934945, 0.03485287, -0.0498423],
                          [-0.13341955, -0.03577899, 0.99041359, -0.08129073],
                          [0., 0., 0., 1.]])
# trans_init = np.eye(4)

max_correspondence_distance = 1
reg_icp = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance, init=initial_guess,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=30000,
        relative_fitness=1e-7,
        relative_rmse=1e-6
    )
)
# print("ICP_Iterative_Closest_Point converged:", reg_icp.converged)
print("Fitness:", reg_icp.fitness)
print("RMSE:", reg_icp.inlier_rmse)
print(f"Transformation Matrix:")
print(reg_icp.transformation)
source_pcd = source.transform(reg_icp.transformation)
o3d.visualization.draw_geometries([source_pcd, target], window_name="After ICP_Iterative_Closest_Point Registration",
                                  width=1024, height=768)
