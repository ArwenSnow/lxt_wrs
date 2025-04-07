"""
Author: Yixuan Su
Date: 2024/11/24 22:57
File: generate_random_transform.py
Description:
"""

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation


def generate_random_transform(rot_range, trans_range):
    rot_x = np.random.uniform(-rot_range[0], rot_range[0])
    rot_y = np.random.uniform(-rot_range[1], rot_range[1])
    rot_z = np.random.uniform(-rot_range[2], rot_range[2])
    r = Rotation.from_euler('xyz', [rot_x, rot_y, rot_z])
    rot_matrix = r.as_matrix()

    trans_x = np.random.uniform(-trans_range[0], trans_range[0])
    trans_y = np.random.uniform(-trans_range[1], trans_range[1])
    trans_z = np.random.uniform(-trans_range[2], trans_range[2])
    translation = np.array([trans_x, trans_y, trans_z])

    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = translation
    return transform


def run_icp_with_random_initial_poses(source, target, num_iterations, rot_range, trans_range, max_iteration=30000,
                                      relative_fitness=1e-7,
                                      relative_rmse=1e-6):
    best_transform = None
    min_error = np.inf
    with open("transform_matrices.txt", "w") as f:
        for i in range(num_iterations):
            initial_transform = generate_random_transform(rot_range, trans_range)
            f.write(f"Iteration {i + 1}:\n")
            np.savetxt(f, initial_transform, fmt='%.6f')
            f.write("\n")
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, max_correspondence_distance=0.05,
                init=initial_transform,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration,
                                                                           relative_fitness=relative_fitness,
                                                                           relative_rmse=relative_rmse
                                                                           ))
            print("Fitness:", reg_p2p.fitness)
            print("RMSE:", reg_p2p.inlier_rmse)
            if reg_p2p.inlier_rmse < min_error:
                min_error = reg_p2p.inlier_rmse
                best_transform = reg_p2p.transformation

    return best_transform, min_error


if __name__ == '__main__':
    source = o3d.io.read_point_cloud("source_normalized_Cloud_cropped.ply")
    target = o3d.io.read_point_cloud("target_normalized.ply")
    num_iterations = 100
    rot_range = [np.pi / 2, np.pi / 2, np.pi / 2]
    trans_range = [1, 1, 1]
    best_transform, min_error = run_icp_with_random_initial_poses(source, target, num_iterations, rot_range,
                                                                  trans_range, max_iteration=30000,
                                                                  relative_fitness=1e-7,
                                                                  relative_rmse=1e-6)
    print("最佳变换矩阵:")
    print(best_transform)
    print("最小配准误差:", min_error)
    # # print("ICP_Iterative_Closest_Point converged:", reg_icp.converged)
    # print("Fitness:", reg_icp.fitness)
    # print("RMSE:", reg_icp.inlier_rmse)
    # print(f"Transformation Matrix:")
    # print(reg_icp.transformation)
    source_pcd = source.transform(best_transform)
    o3d.visualization.draw_geometries([source_pcd, target],
                                      window_name="After ICP_Iterative_Closest_Point Registration",
                                      width=1024, height=768)
