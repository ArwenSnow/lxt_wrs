"""
Author: Yixuan Su
Date: 2025/03/24 20:39
File: abb_partialCP_all.py
Description:

"""

import numpy as np
import trimesh
import abb_trimeshwraper as tw
import modeling.geometric_model as gm
import abb_humath as hm
import open3d as o3d
from panda3d.core import TextNode
from direct.gui.OnscreenText import OnscreenText
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import vision.depth_camera.pcd_data_adapter as vdda
import os

import matplotlib.pyplot as plt


def updateid(idlist, face):
    newid0 = idlist.index(face[0])
    newid1 = idlist.index(face[1])
    newid2 = idlist.index(face[2])
    return [newid0, newid1, newid2]


def generate_colormap_color(index, total_colors=20):
    # cmap = plt.cm.get_cmap('tab20', total_colors)
    cmap = plt.colormaps['tab20']
    return cmap(index % total_colors)


if __name__ == '__main__':
    base = wd.World(cam_pos=[2.01557, 0.637317, 1.88133], w=1280, h=720, lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    icosphere = gm.gen_sphere(radius=0.15, rgba=[0, 0, 1, 0.1], subdivisions=0)
    sample = icosphere.objtrm.vertices

    for pnt in sample:
        gm.gen_sphere(pnt, 0.003, [0, 1, 0, 1]).attach_to(base)
    icosphere.set_rgba([0, 1, 1, 0.1])
    icosphere.attach_to(base)
    name = "rack_5ml_new.stl"
    obj = tw.TrimeshHu(r"E:\ABB-Project\ABB_wrs\suyixuan\ABB\Pose_Estimation\Task5_ICP_GOFA5\meshes", name, scale=0.1)

    mesh = obj.outputTrimesh
    testmesh = gm.GeometricModel(mesh)
    testmesh.attach_to(base)
    gm.gen_sphere(sample[1], 0.01).attach_to(base)
    # base.run()
    for i, pnt in enumerate(sample):
        origin = np.array(pnt)
        intersector = trimesh.base.ray.ray_pyembree.RayMeshIntersector(mesh)
        faces = mesh.faces
        vertices = mesh.vertices
        check_list = []
        origin_list = []
        for face in faces:
            points = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
            direction = np.array(hm.centerPoint(points) - origin)
            check_list.append(direction)
            origin_list.append(origin)

        viewed = intersector.intersects_first(ray_origins=origin_list, ray_directions=check_list)
        viewed = hm.listnorepeat(viewed)
        viewed_faces = [faces[i] for i in viewed]
        list_viewedvertexid = list(set(np.asarray(viewed_faces).flatten().tolist()))

        viewed_vertices = []
        for item in list_viewedvertexid:
            viewed_vertices.append(vertices[item])
        viewed_faces = [updateid(list_viewedvertexid, faces[i]) for i in viewed]
        viewedmesh = trimesh.Trimesh(vertices=viewed_vertices, faces=viewed_faces)
        filename = f"visible_faces_{i}.stl"
        output_path = os.path.join(this_dir, "data", filename)
        viewedmesh.export(output_path)
        print(f"Saved {filename}")
        break

    origin = np.array(sample[1])
    # base.run()

    # intersector = trimesh.base.ray.ray_pyembree.RayMeshIntersector(mesh)
    # faces = mesh.faces
    # vertices = mesh.vertices
    # check_list = []
    # origin_list = []
    #
    # for face in faces:
    #     points = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
    #     direction = np.array(hm.centerPoint(points) - origin)
    #     check_list.append(direction)
    #     origin_list.append(origin)
    #

    # viewed = intersector.intersects_first(ray_origins=origin_list, ray_directions=check_list)
    # viewed = hm.listnorepeat(viewed)
    #

    # viewed_faces = [faces[i] for i in viewed]
    # list_viewedvertexid = list(set(np.asarray(viewed_faces).flatten().tolist()))
    #
    # viewed_vertices = []
    # for item in list_viewedvertexid:
    #     viewed_vertices.append(vertices[item])
    #

    # viewed_faces = [updateid(list_viewedvertexid, faces[i]) for i in viewed]
    #

    # viewedmesh = trimesh.Trimesh(vertices=viewed_vertices, faces=viewed_faces)
    # viewedmesh.export("temp.stl")
    #

    # test = gm.GeometricModel("temp.stl")
    # test.set_rgba([0, 1, 0, 1])
    # test.attach_to(base)
    # base.run()

    viewedmesh_o3d = o3d.io.read_triangle_mesh(
        r"E:\ABB-Project\ABB_wrs\suyixuan\ABB\Pose_Estimation\Task4_Simulated_Viewpoint_icosphere\temp.stl")
    viewedmesh_o3d.compute_vertex_normals()
    pcd = viewedmesh_o3d.sample_points_poisson_disk(number_of_points=4000)
    pcd_np = vdda.o3dpcd_to_parray(pcd)
    gm.gen_pointcloud(pcd_np, pntsize=5).attach_to(base)


    # base.run()
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
