"""Author: Yixuan Su
Date: 2024/11/19 10:23
File: ICP_initial_guess_and_R_xyzt.py
"""
import datetime
import numpy as np
import trimesh
import modeling.geometric_model as gm
import abb_humath as hm
import open3d as o3d
from panda3d.core import TextNode
from direct.gui.OnscreenText import OnscreenText
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import os
from scipy.spatial.transform import Rotation


def updateid(idlist, face):
    newid0 = idlist.index(face[0])
    newid1 = idlist.index(face[1])
    newid2 = idlist.index(face[2])
    return [newid0, newid1, newid2]


#
# def process_icosphere_vertex(sample):
#     for i, pnt in enumerate(sample):
#         origin = np.array(pnt)
#
#         intersector = trimesh.base.ray.ray_pyembree.RayMeshIntersector(mesh)
#         faces = mesh.faces
#         vertices = mesh.vertices
#         check_list = []
#         origin_list = []
#
#         for face in faces:
#             points = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
#             direction = np.array(hm.centerPoint(points) - origin)
#             check_list.append(direction)
#             origin_list.append(origin)
#         viewed = intersector.intersects_first(ray_origins=origin_list, ray_directions=check_list)
#         viewed = hm.listnorepeat(viewed)
#         viewed_faces = [faces[i] for i in viewed]
#         list_viewedvertexid = list(set(np.asarray(viewed_faces).flatten().tolist()))
#         viewed_vertices = []
#         for item in list_viewedvertexid:
#             viewed_vertices.append(vertices[item])
#         viewed_faces = [updateid(list_viewedvertexid, faces[i]) for i in viewed]
#         viewedmesh = trimesh.Trimesh(vertices=viewed_vertices, faces=viewed_faces)
#         filename = f"visible_faces_{i}.stl"
#         output_path = os.path.join(this_dir, "data", filename)
#         viewedmesh.export(output_path)
#         print(f"Saved {filename}")
#         break
#
#     print("i:", i + 1)
#     return i


def perform_icp_registration(source, target, initial_guess=np.eye(4), max_correspondence_distance=1,
                             max_iteration=30000, relative_fitness=1e-7, relative_rmse=1e-6):
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, init=initial_guess,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration,
            relative_fitness=relative_fitness,
            relative_rmse=relative_rmse
        )
    )
    print("Fitness:", reg_icp.fitness)
    print("RMSE:", reg_icp.inlier_rmse)
    print(f"Transformation Matrix:")
    print(reg_icp.transformation)
    source_pcd = source.transform(reg_icp.transformation)
    return reg_icp, source_pcd


def stl_to_pointcloud(stl_file, number_of_points=20000, scale=1000):
    mesh = o3d.io.read_triangle_mesh(stl_file)
    mesh.scale(scale, center=mesh.get_center())
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    point_num = np.asarray(pcd.points).shape[0]
    return pcd, point_num


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


if __name__ == '__main__':
    base = wd.World(cam_pos=[2.01557, 0.637317, 1.88133], w=1280, h=720, lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    output_txt_file = os.path.join(this_dir, "data", "icp_results.txt")
    source_file = os.path.join(this_dir, "target_normalized.ply")

    # name = "rack_5ml_new.stl"
    # mesh = trimesh.load_mesh(os.path.join(this_dir, "meshes", name))
    #
    # icosphere = trimesh.creation.icosphere(radius=0.15, subdivisions=0)
    # sample = icosphere.vertices
    # process_icosphere_vertex(sample)

    source = o3d.io.read_point_cloud("source_normalized_Cloud_cropped.ply")
    num1 = np.asarray(source.points).shape[0]
    print(num1)
    source.paint_uniform_color([1, 0, 0])
    print("显示源点云...")
    o3d.visualization.draw_geometries([source], window_name="source Registration", width=1024, height=768)

    # target = o3d.io.read_point_cloud(r"target_normalized.ply")
    # num2 = np.asarray(target.points).shape[0]
    # print(num2)
    # target.paint_uniform_color([0, 1, 0])
    # print("显示目标点云...")
    # o3d.visualization.draw_geometries([target], window_name="target Registration", width=1024, height=768)

    initial_guess = np.eye(4)
    # initial_guess = np.array([[0.830, 0.113, -0.538, -17.107],
    #                           [-0.548, 0.081, -0.828, -14.283],
    #                           [-0.051, 0.986, 0.130, -3.566],
    #                           [0, 0, 0, 1]])

    best_fitness = -1.0
    best_rmse = float('inf')
    best_transformation = None
    best_stl_file = None
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(output_txt_file, "w", encoding="utf-8") as f:
        f.write(f"ICP 配准结果 - {dt_string}\n")
        f.write(f"点云采样数: {num1}\n")
        f.write(f"初始变换矩阵:\n{initial_guess}\n")
        f.write("--------------------------------------------------\n")

        stl_dir = os.path.join(this_dir, "data")
        stl_files = sorted(os.listdir(stl_dir))
        for filename in os.listdir(stl_dir):
            if filename.endswith(".stl"):
                stl_file = os.path.join(stl_dir, filename)
                print(f"正在处理 STL 文件: {stl_file}")

                stl_pcd, point_num = stl_to_pointcloud(stl_file, number_of_points=num1)
                reg_icp, source_pcd = perform_icp_registration(source, stl_pcd, initial_guess=initial_guess)
                o3d.visualization.draw_geometries([source_pcd, stl_pcd],
                                                  window_name=f"After ICP Registration with {filename}",
                                                  width=1024, height=768)
                fitness = reg_icp.fitness
                rmse = reg_icp.inlier_rmse
                transformation = reg_icp.transformation

                f.write(f"源点云文件: {source_file}\n")
                f.write(f"STL 文件目录: {stl_dir + filename}\n")
                f.write(f"STL点云的点数: {point_num}\n")
                f.write(f"Fitness: {fitness}\n")
                f.write(f"RMSE: {rmse}\n")
                f.write(f"Transformation Matrix:\n{transformation}\n")
                f.write("--------------------------------------------------\n")

                if fitness > best_fitness or (fitness == best_fitness and rmse < best_rmse):
                    best_fitness = fitness
                    best_rmse = rmse
                    best_transformation = transformation
                    best_stl_file = stl_file

                o3d.visualization.draw_geometries([source_pcd, stl_pcd],
                                                  window_name=f"After ICP Registration with {filename}",
                                                  width=1024, height=768)


    def update(textNode, count, task):
        if textNode[0] is not None:
            textNode[0].detachNode()
            textNode[1].detachNode()
            textNode[2].detachNode()
            cam_pos = base.cam.getPos()

            textNode[0] = OnscreenText(
                text=str(cam_pos[0])[0:5],
                fg=(1, 0, 0, 1),
                pos=(1.0, 0.8),
                align=TextNode.ALeft
            )
            textNode[1] = OnscreenText(
                text=str(cam_pos[1])[0:5],
                fg=(0, 1, 0, 1),
                pos=(1.3, 0.8),
                align=TextNode.ALeft
            )
            textNode[2] = OnscreenText(
                text=str(cam_pos[2])[0:5],
                fg=(0, 0, 1, 1),
                pos=(1.6, 0.8),
                align=TextNode.ALeft
            )
            blue_ball = gm.GeometricModel(gm.gen_sphere(radius=0.05, rgba=[0, 0, 1, 1]))
            blue_ball.attach_to(base)
            blue_ball.set_pos(cam_pos)
            return task.again
        cam_view_text = OnscreenText(
            text="Camera View: ",
            fg=(0, 0, 0, 1),
            pos=(1.15, 0.9),
            align=TextNode.ALeft
        )
        testNode = [None, None, None]
        count = [0]
        taskMgr.doMethodLater(0.01, update, "update_cam_pos", extraArgs=[testNode, count], appendTask=True)
        base.run()
