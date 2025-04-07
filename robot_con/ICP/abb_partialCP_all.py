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


def process_icosphere_vertex(sample):
    for i, pnt in enumerate(sample):
        origin = np.array(pnt)
        # 创建射线相交器
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

    print("i:", i + 1)
    return i


if __name__ == '__main__':
    this_dir, this_filename = os.path.split(__file__)
    name = "rack_5ml_new.stl"
    mesh = trimesh.load_mesh(os.path.join(this_dir, "meshes", name))
    icosphere = trimesh.creation.icosphere(radius=0.15, subdivisions=0)
    sample = icosphere.vertices
    process_icosphere_vertex(sample)
