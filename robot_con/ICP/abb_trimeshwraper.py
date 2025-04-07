import copy
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.collision_model as cm
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import numpy as np
import basis.robot_math as rm
import modeling.geometric_model as gm
import os
import pickle
import basis.trimesh as trimeshWan
import trimesh as trimesh
from panda3d.core import NodePath


class TrimeshHu(object):
    """
    TrimeshHu类用于处理三维网格模型，包括加载、修复网格数据、
    更新网格信息、缩放等功能。
    """

    def __init__(self, meshpath=None, name=None, mesh=None, scale=1.0):
        """
        初始化TrimeshHu对象。

        如果传入的mesh对象为空，加载指定路径下的网格数据文件，并修复网格的法线。
        如果提供了mesh对象，则直接使用该对象。

        参数：
        meshpath (str): 网格文件路径。
        name (str): 网格文件名。
        mesh (Trimesh): 现有的Trimesh网格对象。
        scale (float): 网格的缩放比例。
        """
        if not mesh:
            self.name = name
            self.meshpath = meshpath + '/' + name
            self.mesh = trimesh.load(self.meshpath)  # 加载网格文件
        else:
            self.mesh = mesh

        # 修复网格法线
        self.mesh.fix_normals()

        # 更新网格的基本信息
        self.__infoUpdate(self.mesh)

        # 创建原始网格，并根据scale进行缩放
        self.originalmesh = trimesh.Trimesh(
            vertices=self.vertices * scale,
            faces=self.faces * scale,
            face_normals=self.face_normals,
            vertex_normals=self.vertex_normals)

        # 创建原始网格，并根据scale进行缩放
        self.mesh = copy.copy(self.originalmesh)
        self.scalerate = 50
        self.blocksize = self.set_blocksize(self.scalerate)

    def get_bb(self, show=True, option="aabb"):
        """
        获取并返回网格的边界框（bounding box）。
        参数：
        show (bool): 是否展示边界框。
        option (str): 选择显示边界框的类型，默认为“AABB”。
        返回：
        to_origin (np.array): 变换矩阵。
        extents (list): 网格的边界尺寸。
        """
        # 获取网格的边界框的尺寸以及变换矩阵
        extents, to_origin = trimesh.bounds.to_extents(self.mesh.bounds)

        if show:
            if option == "AABB":
                # 使用AABB类型展示边界框
                gm.gen_frame_box(extents, to_origin).attach_to(base)
            if option == "o":
                # 使用变换后的坐标系展示边界框
                gm.gen_frame_box(extents, np.linalg.inv(to_origin)).attach_to(base)
            if option == "max":
                # 使用最大尺寸展示边界框
                extents = [np.max(extents) * self.blocksize] * 3
                gm.gen_frame_box(extents, to_origin).attach_to(base)

        # 将变换矩阵保存到类的属性中
        self.boxhomo = to_origin

        return to_origin, extents

    def set_blocksize(self, rate=50):
        """
        设置网格的块大小。
        根据网格的边界框尺寸和传入的比例，计算块大小。

        参数：
        rate (int): 块大小的比例。

        返回：
        blocksize (float): 计算出的块大小。
        """

        # 获取网格的边界框尺寸
        _, extents = self.get_bb(show=True)

        # 根据最大边界尺寸和比例计算块大小
        blocksize = np.max(extents) / rate
        return blocksize

    @property
    def get_blocksize(self):
        """
        获取当前的块大小。

        返回：
        float: 当前网格的块大小
        """
        return self.blocksize

    def voxelization(self, voxel, hollow):
        """
           对网格进行体素化处理，并根据是否空心设置不同的处理方式。

           参数：
           voxel (float): 体素化的分辨率。
           hollow (bool): 是否生成空心的体素化模型。如果为True，生成空心模型；否则生成填充模型。

           处理过程：
           - 根据`hollow`参数选择不同的体素化方法：
               - `True`: 生成空心体素化模型。
               - `False`: 生成填充的体素化模型（默认方法是'base'）。
        """

        # 设置体素化分辨率
        self.voxel = voxel

        if hollow == True:  # 如果选择空心体素化
            # 生成空心体素化模型，并获取变换矩阵和矩阵信息
            self.voxelizedmodel = self.mesh.voxelized(voxel).hollow()
            self.tfmatrix = self.voxelizedmodel.transform
            self.matrix = self.voxelizedmodel.matrix
            print(self.matrix.shape)  # 打印矩阵的形状
            self.points = self.voxelizedmodel.points  # 获取体素化后的点集
            self.mesh = self.voxelizedmodel.as_boxes()  # 获取体素化后的网格

        # 获取体素化后的网格
        else:  # 如果选择填充体素化
            # 生成填充体素化模型，并获取变换矩阵和矩阵信息
            self.voxelizedmodel = self.mesh.voxelized(voxel).fill(method='base')
            self.tfmatrix = self.voxelizedmodel.transform
            self.matrix = self.voxelizedmodel.matrix
            self.points = self.voxelizedmodel.points  # 获取体素化后的点集
            self.mesh = self.voxelizedmodel.as_boxes()  # 获取体素化后的网格

        # 更新网格信息
        self.__infoUpdate(self.mesh)

        '''
        slide matrix
        '''
        # output_mat = np.zeros([self.scalerate+1,self.scalerate+1,self.scalerate+1,1])
        # mat_int = np.expand_dims(self.matrix.astype(np.int), axis = 3)
        # offset_0 = np.zeros([output_mat.shape[0]-mat_int.shape[0],mat_int.shape[1], mat_int.shape[2], 1])
        # offset_1 = np.zeros([output_mat.shape[0], output_mat.shape[1]-mat_int.shape[1], mat_int.shape[2], 1])
        # offset_2 = np.zeros([output_mat.shape[0], output_mat.shape[1], output_mat.shape[2]-mat_int.shape[2], 1])
        # mat_int = np.concatenate([mat_int, offset_0], axis = 0)
        # mat_int = np.concatenate([mat_int, offset_1], axis = 1)
        # mat_int = np.concatenate([mat_int, offset_2], axis = 2)
        # mat_int[0][0][0][0] = 1000
        # multipler = np.full((self.scalerate+1,self.scalerate+1,self.scalerate+1,1), self.blocksize)
        # output_mat = np.round(mat_int * multipler, decimals=7)
        # # print(mat_int.shape)

    def get_node_matrix(self):
        """
            获取节点变换矩阵。

            该方法计算网格的节点变换矩阵，通过应用体素化的变换矩阵和缩放值
            来获取每个节点的位置。

            返回：
            np.ndarray: 变换后的节点矩阵。
        """
        # 计算每个节点的位置
        matrix = [
            [
                [
                    [
                        self.matrix[i][j][k] * i * self.voxel + self.tfmatrix[0][3],
                        self.matrix[i][j][k] * j * self.voxel + self.tfmatrix[1][3],
                        self.matrix[i][j][k] * k * self.voxel + self.tfmatrix[2][3]
                    ]
                    for k in range(len(self.matrix[i][j]))
                ]
                for j in range(len(self.matrix[i]))
            ]
            for i in range(len(self.matrix))
        ]

        # 将计算结果转换为NumPy数组
        self.node_matrix = np.asarray(matrix)
        return self.node_matrix

    def get_transform(self):
        """
            获取变换矩阵。

            返回：
            np.ndarray: 当前网格的变换矩阵。
        """
        return self.tfmatrix

    def show_balls(self):
        """
            显示体素化网格的球体表示。

            遍历每个体素点并生成球体模型（红色半透明球）来显示网格。
        """
        for point in self.points:
            # 使用球体生成函数显示每个体素的中心位置
            gm.gen_sphere(point, .001, [1, 0, 0, 0.5]).attach_to(base)

    def triWantotri(self, mesh):
        """
            将一个网格对象转换为Trimesh格式。

            该方法将输入网格的面、顶点、面法线和顶点法线提取出来，
            然后重新创建一个Trimesh网格对象。

            参数：
            mesh (Trimesh): 输入的网格对象。

            返回：
            Trimesh: 转换后的Trimesh网格对象。
        """
        # 提取面、顶点、面法线和顶点法线
        faces = np.asarray(mesh.faces)
        vertices = np.asarray(mesh.vertices)
        face_normals = np.asarray(mesh.face_normals)
        vertex_normals = np.asarray(mesh.vertex_normals)

        # 创建新的Trimesh网格
        new_mesh = trimesh.Trimesh(
            faces=faces,
            vertices=vertices,
            face_normals=face_normals,
            vertex_normals=vertex_normals
        )
        return new_mesh

    def show_hited_balls(self, observe_origin, target, shape="box", generateCKlist=False):
        """
            显示被命中的体素点，并根据给定的形状在目标上生成可视化对象。

            参数：
            observe_origin (list or np.ndarray): 观察的原点位置。
            target (object): 目标对象，用于显示生成的元素。
            shape (str): 可选的形状类型，默认为 "box"，也可以是 "sphere"。
            generateCKlist (bool): 是否生成并返回包含所有命中的体素元素的列表。

            返回：
            list: 如果generateCKlist为True，则返回一个包含所有体素元素的列表。
         """

        # 获取被命中的体素点
        hited_list = self.hitray(observe_origin)

        # 移除离群点
        def display_inlier_outlier(cloud, ind):
            """
               显示离群点和内点的可视化效果。

               参数：
               cloud (PointCloud): 输入的点云数据。
               ind (list): 点云数据的索引。
            """
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])  # 离群点显示为红色
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # 内点显示为灰色
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                              zoom=0.3412,
                                              front=[0.4257, -0.2125, -0.8795],
                                              lookat=[0, 0, 0],
                                              up=[-0.0694, -0.9768, 0.2024])

        # 将命中的点转换为Open3D点云并移除离群点
        pcd = vdda.nparray_to_o3dpcd(np.asarray(hited_list))
        cl, ind = pcd.remove_radius_outlier(nb_points=15, radius=self.blocksize * 3)
        display_inlier_outlier(pcd, ind)

        # 更新命中点列表
        hited_list = vdda.o3dpcd_to_parray(cl)

        # 创建局部网格并导出为STL文件
        partialmesh = self.triWantotri(
            gm.gen_box(extent=[self.voxel] * 3, homomat=rm.homomat_from_posrot(hited_list[0])).objtrm)
        partialmesh.export("partial_bunny.stl")

        partialmesh_cklist = []
        center = np.average(hited_list, axis=0)  # 计算命中点的平均位置作为中心

        # 遍历命中点，根据指定形状生成对应的可视化对象
        for point in hited_list:
            if shape == "box":
                # 生成体素箱子
                element = cm.gen_box(extent=[self.voxel * 1.2] * 3, homomat=rm.homomat_from_posrot(point),
                                     rgba=[1, 0, 0, 0.1])
                element.attach_to(target)  # 将元素附加到目标上
                if generateCKlist:
                    partialmesh_cklist.append(element)  # 如果需要，加入到返回列表中
            else:
                # 生成球体
                gm.gen_sphere(point, .001, [1, 0, 0, 0.5]).attach_to(target)

        # 如果需要，返回包含所有生成元素的列表
        if generateCKlist:
            return partialmesh_cklist

    def cpt_briefgrasp(self, observe_origin, target, gripper, grasp_info_list, generateCKlist=True):
        """
        计算和展示在目标物体上的简短抓取尝试。

        参数：
        observe_origin (list or np.ndarray): 观察的原点位置。
        target (object): 目标对象，用于展示生成的元素。
        gripper (object): 抓取器对象，用于执行抓取动作。
        grasp_info_list (list): 抓取信息列表，每个元素包含抓取的相关参数。
        generateCKlist (bool): 是否生成并返回包含所有有效抓取的列表。

        返回：
        None: 方法无返回值，但会在目标对象上生成可视化元素。
        """
        # 获取命中的体素元素列表
        partialmeshCKlist = self.show_hited_balls(observe_origin, target, shape="box", generateCKlist=generateCKlist)

        # 初始化抓取列表和方向列表
        briefgrasplist = []
        voxel_grct_list = []
        direction_z_list = []
        direction_x_list = []

        # 遍历抓取信息列表
        for i, grasp_info in enumerate(grasp_info_list):
            # 每5次抓取信息尝试一次
            if i % 5 == 0:

                jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info

                # 执行抓取器的抓取操作
                gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)

                # 生成一个"stick"模型，表示抓取器的握紧区域
                checker = cm.gen_stick(spos=jaw_center_pos + 0.11 * jaw_center_rotmat[:, 0],
                                       epos=jaw_center_pos - 0.11 * jaw_center_rotmat[:, 0],
                                       rgba=(0.1, 0.1, 0.1, 0.1), thickness=0.001)

                # 计算抓取点的体素位置和方向
                voxel_grct = np.around(jaw_center_pos / self.blocksize, decimals=0, out=None)
                direction_z = hnd_rotmat[:, 2]
                direction_x = hnd_rotmat[:, 0]
                voxel_grct_list.append(voxel_grct)
                direction_z_list.append(direction_z)
                direction_x_list.append(direction_x)

                # 检查抓取器是否与目标碰撞
                collideinfo = checker.is_mcdwith(partialmeshCKlist, toggle_contacts=True)

                if not collideinfo[0]:  # 如果没有碰撞
                    briefgrasplist.append(grasp_info)  # 添加到有效抓取列表

                    # 可视化抓取器的抓取区域（表示为箭头）
                    gm.gen_arrow(spos=jaw_center_pos - jaw_width * jaw_center_rotmat[:, 0] / 2,
                                 epos=jaw_center_pos + jaw_width * jaw_center_rotmat[:, 0] / 2,
                                 rgba=(0.1, 0.1, 0.1, 0.01), thickness=0.002).attach_to(base)

                else:  # 如果发生碰撞
                    # 可视化碰撞区域
                    pass

        # 返回None表示方法没有返回值
        return None

    def hitray(self, observe_origin=[.0, .0, -.09]):
        """
            使用射线与网格进行交互，找出所有与射线相交的网格点。

            参数：
            observe_origin (list or np.ndarray): 观察点的位置，默认是[0.0, 0.0, -0.09]。

            返回：
            np.ndarray: 返回所有与射线相交的点，形状为(n, 3)，其中n为交点的数量。
        """
        # 创建RayMeshIntersector对象，用于检测射线与网格的交点
        checker = trimesh.base.ray.ray_pyembree.RayMeshIntersector(self.mesh)

        # 将观察点设置为输入的observe_origin，转换为numpy数组
        observation = np.array([observe_origin])

        # 计算射线方向，射线从观察点指向每个网格点
        ray_directions = [point - observation[0] for point in self.points]
        ray_origins = [observation[0]] * len(self.points)

        # 使用射线与网格的交点检测
        hitinfo = checker.intersects_id(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False,
                                        max_hits=1,
                                        return_locations=True)

        # 初始化一个列表存储所有与射线相交的点
        hited_list = []

        # 遍历所有的网格点，检查每个点是否与射线相交
        for i, point in enumerate(self.points):
            if np.linalg.norm(point - hitinfo[2][i]) <= self.voxel:  # 如果点到交点的距离小于体素大小，认为是命中
                hited_list.append(point)

        # 将结果转换为numpy数组
        hited_list = np.asarray(hited_list)

        # 返回所有命中的点
        return hited_list

    @property
    def outputTrimesh(self):
        """
            将当前网格数据（包括顶点、面、法线等）转化为Trimesh对象。

            返回：
                Trimesh: 返回一个新的Trimesh对象。
        """
        self.newmesh = trimeshWan.Trimesh(vertices=self.vertices, faces=self.faces, face_normals=self.face_normals,
                                          vertex_normals=self.vertex_normals)
        return self.newmesh

    def __infoUpdate(self, mesh):
        """
            更新网格的基本信息，包括顶点、面、法线等。

            参数：
                mesh (Trimesh): 传入的Trimesh网格对象。
        """
        self.faces = np.asarray(mesh.faces)  # 获取网格的面
        self.vertices = np.asarray(mesh.vertices)  # 获取网格的面
        self.face_normals = np.asarray(mesh.face_normals)  # 获取网格的面法线
        self.vertex_normals = np.asarray(mesh.vertex_normals)  # 获取网格的面法线

    def meshTransform(self, rotaxis=np.array([0, 0, 1]), angle=np.radians(90), translation=np.array([0, 0, 0])):
        """
            对网格进行变换，包括旋转和平移。

            参数：
                rotaxis (np.ndarray): 旋转轴，默认沿Z轴旋转。
                angle (float): 旋转角度，默认90度。
                translation (np.ndarray): 平移向量，默认无平移。
        """
        rotmat = rm.rotmat_from_axangle(rotaxis, angle)  # 计算旋转矩阵
        homomate = rm.homomat_from_posrot(translation, rotmat)  # 计算平移和旋转的齐次变换矩阵
        self.mesh.apply_transform(homomate)  # 应用变换矩阵到网格

        # 可选的手动更新网格的顶点和法线（但这里已通过apply_transform自动完成）
        # self.vertices = np.asarray([rotmat.dot(vert) + translation for vert in self.mesh.vertices])
        # self.faces = self.mesh.faces
        # self.face_normals =  np.asarray([rotmat.dot(face_normal) + translation for face_normal in self.mesh.face_normals])
        # self.vertex_normals =  np.asarray([rotmat.dot(vertex_normal) + translation for vertex_normal in self.mesh.vertex_normals])
        # self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, face_normals=self.face_normals,
        #                                vertex_normals=self.vertex_normals)

    def export(self, outputfile, outputname):
        """
            导出网格为STL文件。

            参数：
                outputfile (str): 输出路径，如果为空则默认为当前目录。
                outputname (str): 输出文件名。
        """
        this_dir, this_filename = os.path.split(__file__)  # 获取当前脚本文件所在的目录
        extention = ".stl"  # 默认的输出文件扩展名为.stl

        # 如果未指定输出路径，则使用当前目录
        if outputfile:
            outputfile = outputfile
        else:
            outputfile = this_dir

        # 导出网格为STL文件
        self.mesh.export(outputfile + "/" + outputname + extention)


if __name__ == '__main__':

    # 初始化仿真世界并设置相机位置
    base = wd.World(cam_pos=[2.01557, 0.637317, 1.88133], w=960, h=540, lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)  # 生成并附加参考坐标系框架到世界
    this_dir, this_filename = os.path.split(__file__)  # 获取当前文件目录和文件名

    name = "armadillo.stl"  # 设置STL文件名称
    mesh = TrimeshHu("./3dcnnobj/", name)  # 加载3D模型

    # 以下为注释掉的代码，可以用来生成盒子并设置缩放
    # name = "bo"
    # box = gm.gen_box([.090,.090,.090]).objtrm
    # mesh = TrimeshHu(mesh = box, scale=1)
    # mesh.set_scale((0.001, 0.001, 0.001))
    # mesh.voxelization(45, hollow = False)

    # 生成球体并设置其颜色和细分级别
    icosphere = gm.gen_sphere(radius=0.2, rgba=[0, 0, 1, 0.1], subdivisions=1)
    sample = icosphere.objtrm.vertices  # 提取球体顶点
    icosphere.attach_to(base)  # 将球体添加到世界中

    # 导入Open3D、点云数据处理和接触对模块
    import open3d as o3d
    import vision.depth_camera.pcd_data_adapter as vdda
    import freeholdcontactpairs as f

    # 加载相机标定矩阵和点云数据
    calib_mat = pickle.load(open("phoxi_calibmat.pkl", "rb"))
    pcd_list = pickle.load(open("pc/bunnypcd.pkl", "rb"))[2]

    # 使用标定矩阵对点云进行坐标变换
    pcd_list = rm.homomat_transform_points(calib_mat, pcd_list)
    n_pcd_list = []
    for pcd in pcd_list:
        if pcd[2] > 810 and pcd[2] < 930 and pcd[1] > 100 and pcd[1] < 270 and pcd[0] > 700 and pcd[0] < 930:
            n_pcd_list.append(pcd)  # 过滤符合条件的点云

    # 将点云数据转为Open3D格式并去除离群点
    pcd = vdda.nparray_to_o3dpcd(np.asarray(n_pcd_list))
    pcd, ind = pcd.remove_radius_outlier(nb_points=15, radius=5)
    o3d.visualization.draw_geometries([pcd])  # 可视化点云

    # 使用Alpha Shape算法从点云创建网格
    alpha = 20
    mmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    radii = np.array([0.005, 0.01, 0.02, 0.04])
    # mmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mmesh.compute_vertex_normals()  # 计算顶点法线
    o3d.visualization.draw_geometries([mmesh], mesh_show_back_face=True)  # 显示网格

    # 将Open3D网格转换为Trimesh对象并处理
    mmesh_trimesh = vdda.o3dmesh_to_trimesh(mmesh)
    p_obj = cm.CollisionModel(mmesh_trimesh)
    p_obj.set_scale((0.001, 0.001, 0.001))
    p_obj.set_rgba((0.5, 0.5, 0.5, 1))  # 设置物体颜色
    p_obj.attach_to(base)  # 将物体添加到仿真世界

    # 导出部分网格为STL文件并显示接触面
    mmesh_trimesh.export("partial_bunny.stl")
    partial = f.FreeholdContactpairs("partial_bunny.stl")
    partial.showallFaces()  # 显示所有接触面

    base.run()  # 启动仿真

    # 创建TrimeshHu对象并进行体素化
    mmesh_hu = TrimeshHu(mesh=mmesh_trimesh)
    mmesh_hu.voxelization(3, hollow=True)
    mmesh_hu.get_node_matrix()
    mmesh_hu.get_transform()

    # 创建CollisionModel对象并设置其属性
    t = cm.CollisionModel(mmesh_hu.outputTrimesh)
    t.set_scale((0.001, 0.001, 0.001))
    t.set_rgba((0, 1, 0, .11))  # 设置颜色为绿色
    t.attach_to(base)  # 将其添加到仿真世界

    base.run()  # 再次启动仿真

    # pctrim = vdda.o3dmesh_to_trimesh(pcd)  # 将点云转换为Trimesh对象，已注释
    radii = [0.005, 0.01, 0.02, 0.04]  # 定义球半径
    cl, ind = pcd.remove_radius_outlier(nb_points=15, radius=0.003)  # 移除离群点
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        radii))  # 使用Ball Pivoting算法生成网格

    # objpdnp_raw.reparentTo(self._objpdnp)  # 注释掉的代码

    # 创建几何模型并缩放
    a = gm.GeometricModel(pcd)
    a.set_scale((0.001, 0.001, 0.001))
    a.attach_to(base)  # 将几何模型添加到世界

    base.run()  # 启动仿真

    # 对mesh进行变换操作
    mesh.meshTransform(rotaxis=np.array([0, 0, 1]), angle=np.radians(45), translation=np.array([0, 0, 0]))
    mesh.voxelization(.0045, hollow=True)  # 体素化网格
    mesh.get_node_matrix()  # 获取节点矩阵
    mesh.get_transform()  # 获取变换矩阵

    # 导出网格并显示
    mesh.export(this_dir, "box_vox")
    c = cm.CollisionModel(mesh.outputTrimesh)
    # c.set_scale((0.001, 0.001, 0.001))
    c.set_rgba((0, 1, 0, .11))  # 设置颜色为绿色
    c.attach_to(base)  # 将网格添加到仿真世界

    objNode = [None]  # 初始化对象节点
    voxelNode = [None]  # 初始化体素节点
    observeNode = [None]  # 初始化观察节点


    def update(textNode, objNode, voxelNode, observeNode, count, task):
        # 检查并移除之前的观察节点
        if observeNode[0] is not None:
            observeNode[0].detachNode()

        # 创建一个新的观察节点
        observeNode[0] = NodePath("observe")

        # 使用当前计数器（count[0]）显示与样本相关的球体
        mesh.show_hited_balls(observe_origin=sample[count[0]], target=observeNode[0])

        # 生成一个球体并附加到观察节点上
        gm.gen_sphere(sample[count[0]]).attach_to(observeNode[0])

        # 将观察节点附加到渲染图形中
        observeNode[0].reparent_to(base.render)

        # 更新计数器
        count[0] += 1

        # 更新文本节点显示当前相机位置
        if textNode[0] is not None:
            textNode[0].detachNode()
            textNode[1].detachNode()
            textNode[2].detachNode()

        # 获取相机的当前坐标
        cam_pos = base.cam.getPos()

        # 创建并显示相机的X、Y、Z坐标
        textNode[0] = OnscreenText(
            text=str(cam_pos[0])[0:5],
            fg=(1, 0, 0, 1),  # 红色
            pos=(1.0, 0.8),
            align=TextNode.ALeft
        )
        textNode[1] = OnscreenText(
            text=str(cam_pos[1])[0:5],
            fg=(0, 1, 0, 1),  # 绿色
            pos=(1.3, 0.8),
            align=TextNode.ALeft
        )
        textNode[2] = OnscreenText(
            text=str(cam_pos[2])[0:5],
            fg=(0, 0, 1, 1),  # 蓝色
            pos=(1.6, 0.8),
            align=TextNode.ALeft
        )

        # 返回任务继续执行
        return task.again


    # 在屏幕上显示相机视角文字
    cam_view_text = OnscreenText(
        text="Camera View: ",
        fg=(0, 0, 0, 1),
        pos=(1.15, 0.9),
        align=TextNode.ALeft
    )

    # 初始化节点和计数器
    testNode = [None, None, None]
    count = [0]

    # 使用定时器每秒调用一次update函数
    taskMgr.doMethodLater(1, update, "addobject", extraArgs=[testNode, objNode, voxelNode, observeNode, count],
                          appendTask=True)

    # 启动仿真
    base.run()
