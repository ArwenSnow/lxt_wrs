from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.collision_model as cm
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import modeling.geometric_model as gm
import os
import open3d as o3d
import vision.depth_camera.pcd_data_adapter as vdda

if __name__ == '__main__':
    base = wd.World(cam_pos=[0.2001557, 0.0637317, 0.1088133], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    # gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    Mesh = o3d.io.read_triangle_mesh("kit_model/Amicelli_800_tex.obj")
    Mesh.compute_vertex_normals()
    pcd1 = Mesh.sample_points_poisson_disk(number_of_points=5000)
    pcd1, ind = pcd1.remove_radius_outlier(nb_points=50, radius=5)
    pcd1_np = vdda.o3dpcd_to_parray(pcd1)
    gm.gen_pointcloud(pcd1_np).attach_to(base)
    base.run()

    radii = [3, 3, 3, 3]
    mmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd1, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([pcd, mmesh])

    mmesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mmesh], mesh_show_back_face=True)
    mmesh_trimesh = vdda.o3dmesh_to_trimesh(mmesh)

    mmesh_trimesh.export("edgenooltst.stl")
    # obj_mmesh = cm.CollisionModel('edgenool.stl')
    obj_mmesh = cm.CollisionModel('edge.stl')
    obj_mmesh.set_rgba((0, 1, 1, 1))
    obj_mmesh.attach_to(base)

    base.run()


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
            align=TextNode.ALeft)
        textNode[1] = OnscreenText(
            text=str(cam_pos[1])[0:5],
            fg=(0, 1, 0, 1),
            pos=(1.3, 0.8),
            align=TextNode.ALeft)
        textNode[2] = OnscreenText(
            text=str(cam_pos[2])[0:5],
            fg=(0, 0, 1, 1),
            pos=(1.6, 0.8),
            align=TextNode.ALeft)
        return task.again


    cam_view_text = OnscreenText(
        text="Camera View: ",
        fg=(0, 0, 0, 1),
        pos=(1.15, 0.9),
        align=TextNode.ALeft)
    testNode = [None, None, None]
    count = [0]
    taskMgr.doMethodLater(0.01, update, "addobject", extraArgs=[testNode, count],
                          appendTask=True)

    base.run()
