import open3d as o3d
import numpy as np

source_points = np.array([
    [-70.7276611328, 43.5730819702, -24.2528686523],
    [-66.9071655273, -37.76978302, -33.3447875977],
    [40.5988922119, -15.0885686874, -94.3760375977],
    [37.0685577393, 51.939491272, -85.9612426758],
    [55.1555023193, 52.6165542603, -56.0742797852],
    [60.6202850342, -27.1304092407, -64.1505737305]
])

target_points = np.array([
    [-58.434009552, 47.5, 40.8018951416],
    [-57.2034492493, 47.5, -41.7822799683],
    [65, 40.4656677246, -36.6255645752],
    [65, 39.0477409363, 35.7997817993],
    [64.9049072266, 7.59446191788, 39.0387191772],
    [64.9435653687, 3.50905990601, -40.7071075439]
])


def compute_transform(source_points, target_points):
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    H = np.dot(source_centered.T, target_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t = centroid_target - np.dot(R, centroid_source)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t
    return transform_matrix


initial_transform = compute_transform(source_points, target_points)
print("Initial Transformation Matrix from 6-point matching:")
print(initial_transform)

source = o3d.io.read_point_cloud(r"source_normalized_Cloud_cropped.ply")
target = o3d.io.read_point_cloud(r"target_normalized.ply")
max_correspondence_distance = 1

reg_icp = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance, init=initial_transform,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=30000,
        relative_fitness=1e-7,
        relative_rmse=1e-6
    )
)

print("Fitness:", reg_icp.fitness)
print("RMSE:", reg_icp.inlier_rmse)
print(f"Final Transformation Matrix after Open3D_ICP_Edition:")
print(reg_icp.transformation)
source_pcd = source.transform(reg_icp.transformation)
o3d.visualization.draw_geometries([source_pcd, target], window_name="After Open3D_ICP_Edition Registration", width=1024,
                                  height=768)
