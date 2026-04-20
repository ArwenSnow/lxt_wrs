# 基于几何特征 + 对偶接触（antipodal grasp）条件的接触点搜索方法。
# 完全无可视化、无文件保存的 ContactDetector（内存数据传入/返回）
import numpy as np
import trimesh as trimesh
import itertools
import pyransac3d as pyrsc
import open3d as o3d
import utils.robot_math as rm
import utils.pcd_data_adapter as pda
import utils.trimesh_primitive_generator as tpg


class ContactDetector:
    def __init__(self, objpath, objname,  feature_data=None):
        self.objname = objname
        self.objpath = objpath
        if feature_data is not None:
            self._set_features_from_data(feature_data)
        else:
            self.edge_pnt = np.asarray([])                 # 边上
            self.edge_normal = np.asarray([])
            self.edge_id = np.asarray([])
            self.surface_pnt = np.asarray([])              # 表面
            self.surface_normal = np.asarray([])
            self.surface_id = np.asarray([])
            self.vertex_pnt_clustered = np.asarray([])     # 角点
            self.vertex_normal_clustered = np.asarray([])
            self.vertex_id_clustered = np.asarray([])
            self.pcd = None                                # 点云
            self.pcd_np = np.asarray([])
            self.pcd_id = np.asarray([])
            self.pcd_normal = np.asarray([])
        # detect scale
        if hasattr(self, 'pcd_np') and len(self.pcd_np) > 0:
            self.detect_scale = self.get_pcd_max_scale() * 1.1
        else:
            self.detect_scale = 1.0

    def _set_features_from_data(self, data: dict):
        self.edge_pnt = np.asarray(data.get('edge_pnt', np.asarray([])))
        self.edge_normal = np.asarray(data.get('edge_normal', np.asarray([])))
        self.edge_id = np.asarray(data.get('edge_id', np.asarray([])))

        self.surface_pnt = np.asarray(data.get('surface_pnt', np.asarray([])))
        self.surface_normal = np.asarray(data.get('surface_normal', np.asarray([])))
        self.surface_id = np.asarray(data.get('surface_id', np.asarray([])))

        self.vertex_pnt_clustered = np.asarray(data.get('vertex_pnt_clustered', np.asarray([])))
        self.vertex_normal_clustered = np.asarray(data.get('vertex_normal_clustered', np.asarray([])))
        self.vertex_id_clustered = np.asarray(data.get('vertex_id_clustered', np.asarray([])))

        if data.get('pcd_np', None) is not None:
            self.pcd_np = np.asarray(data['pcd_np'])
            self.pcd = pda.nparray_to_o3dpcd(self.pcd_np)
        elif data.get('pcd', None) is not None:
            self.pcd = data['pcd']
            self.pcd_np = pda.o3dpcd_to_parray(self.pcd)
        else:
            self.pcd = None
            self.pcd_np = np.asarray([])

        self.pcd_id = np.asarray(data.get('pcd_id', np.asarray([])))
        self.pcd_normal = np.asarray(data.get('pcd_normal', np.asarray([])))
        if hasattr(self, 'pcd') and self.pcd is not None:
            self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)       # 找邻居

    def main_detector(self, voxel_size=0.012):
        # 采样，降低点数
        self.surface_sampled, self.surface_sampled_id, self.surface_sampled_normal = self.sample_contact(
            self.surface_pnt, self.surface_id, self.surface_normal, "voxel", voxel_size)
        self.edge_sampled, self.edge_sampled_id, self.edge_sampled_normal = self.sample_contact(
            self.edge_pnt, self.edge_id, self.edge_normal, "voxel", voxel_size)
        self.corner_sampled, self.corner_sampled_id, self.corner_sampled_normal = self.sample_contact(
            self.vertex_pnt_clustered, self.vertex_id_clustered, self.vertex_normal_clustered, "voxel", voxel_size)
        self.full_pcd_sampled, self.full_pcd_sampled_id, _ = self.sample_contact(
            self.pcd_np, self.pcd_id, self.pcd_normal, "voxel", voxel_size)

        # pairing & grasp planning (no collision checking)
        # 找接触点对
        self.pairing_contacts()

        # 生成抓取信息
        grip_info = self.grasp_planning(check_collision=False)
        print("Detected contact pairs:", len(getattr(self, 'pair_pnts', [])))
        return {
            'pair_pnts': getattr(self, 'pair_pnts', []),
            'pair_type': getattr(self, 'pair_type', []),
            'pair_pnts_id': getattr(self, 'pair_pnts_id', []),
            'pair_normals': getattr(self, 'pair_normals', []),
            'pair_contact_vecs': getattr(self, 'pair_contact_vecs', []),
            'grip_info': grip_info
        }

    def sample_contact(self, points, indices, normals, method="voxel", voxel_size=0.012):  # 对点云做降采样
        if points is None or len(points) == 0:
            return [], [], []
        sampled_contact_id = []
        sampled_contact_pnt = []
        sampled_contact_normal = []
        pcd = pda.nparray_to_o3dpcd(points)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        if method == "voxel":
            min_bound = np.min(self.pcd_np, axis=0)
            max_bound = np.max(self.pcd_np, axis=0)
            sampled_contact, contact_id, _ = pda.nparray_to_o3dpcd(
                points).voxel_down_sample_and_trace(voxel_size=voxel_size,
                                                    min_bound=min_bound,
                                                    max_bound=max_bound,
                                                    approximate_class=False)
            sample = pda.o3dpcd_to_parray(sampled_contact)
            for pnt in sample:
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pnt, 1)
                sampled_contact_pnt.append(points[idx[0]])
                sampled_contact_id.append(indices[idx[0]] if len(indices) > 0 else idx[0])
                sampled_contact_normal.append(normals[idx[0]] if len(normals) > 0 else np.array([0, 0, 1]))
        elif method == "uniform":
            sample = pda.o3dpcd_to_parray(pcd.uniform_down_sample(every_k_points=100))
            for pnt in sample:
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pnt, 1)
                sampled_contact_pnt.append(points[idx[0]])
                sampled_contact_id.append(indices[idx[0]] if len(indices) > 0 else idx[0])
                sampled_contact_normal.append(normals[idx[0]] if len(normals) > 0 else np.array([0, 0, 1]))
        else:
            raise ValueError("Unexpected sampling method.")
        return sampled_contact_pnt, sampled_contact_id, sampled_contact_normal

    def pairing_contacts(self):  # 找接触点对
        temp_pair_pnts = []
        temp_pair_type = []
        temp_pair_pnts_id = []
        temp_pair_normals = []
        temp_pair_contact_vecs = []

        # corner → corner，corner → edge，corner → surface
        for i in self.corner_sampled_id[:]:
            detector_mesh = self.get_corner_detector(i)
            anchor_normal = self.pcd_normal[i]
            anchor = self.pcd_np[i]
            cc_pnts, cc_pnts_id, cc_normal, cc_contact_vec = self.find_antipodal(detector_mesh, anchor, i,
                                                                                 anchor_normal, self.corner_sampled,
                                                                                 self.corner_sampled_id,
                                                                                 self.corner_sampled_normal,
                                                                                 threshold=0.95)
            ce_pnts, ce_pnts_id, ce_normal, ce_contact_vec = self.find_antipodal(detector_mesh, anchor, i,
                                                                                 anchor_normal, self.edge_sampled,
                                                                                 self.edge_sampled_id,
                                                                                 self.edge_sampled_normal,
                                                                                 threshold=0.7)
            cf_pnts, cf_pnts_id, cf_normal, cf_contact_vec = self.find_antipodal(detector_mesh, anchor, i,
                                                                                 anchor_normal, self.surface_sampled,
                                                                                 self.surface_sampled_id,
                                                                                 self.surface_sampled_normal,
                                                                                 threshold=0.98)
            c_temp_pair_pnts = [item for item in cc_pnts + ce_pnts + cf_pnts]
            c_temp_pair_type = ['cc'] * len(cc_pnts) + ['ce'] * len(ce_pnts) + ['cf'] * len(cf_pnts)
            c_temp_pair_pnts_id = [item for item in cc_pnts_id + ce_pnts_id + cf_pnts_id]
            c_temp_pair_normals = [item for item in cc_normal + ce_normal + cf_normal]
            c_temp_pair_contact_vecs = [item for item in cc_contact_vec + ce_contact_vec + cf_contact_vec]
            temp_pair_pnts.append(c_temp_pair_pnts)
            temp_pair_type.append(c_temp_pair_type)
            temp_pair_pnts_id.append(c_temp_pair_pnts_id)
            temp_pair_normals.append(c_temp_pair_normals)
            temp_pair_contact_vecs.append(c_temp_pair_contact_vecs)

        # edge → edge，edge → surface
        for i in self.edge_sampled_id[:]:
            detector_mesh = self.get_edge_detector(i)
            anchor_normal = self.pcd_normal[i]
            anchor = self.pcd_np[i]
            ee_pnts, ee_pnts_id, ee_normal, ee_contact_vec = self.find_antipodal(detector_mesh, anchor, i,
                                                                                 anchor_normal, self.edge_sampled,
                                                                                 self.edge_sampled_id,
                                                                                 self.edge_sampled_normal,
                                                                                 threshold=0.95)
            ef_pnts, ef_pnts_id, ef_normal, ef_contact_vec = self.find_antipodal(detector_mesh, anchor, i,
                                                                                 anchor_normal, self.surface_sampled,
                                                                                 self.surface_sampled_id,
                                                                                 self.surface_sampled_normal,
                                                                                 threshold=0.95)
            e_temp_pair_pnts = [item for item in ee_pnts + ef_pnts]
            e_temp_pair_type = ['ee'] * len(ee_pnts) + ['ef'] * len(ef_pnts)
            e_temp_pair_pnts_id = [item for item in ee_pnts_id + ef_pnts_id]
            e_temp_pair_normals = [item for item in ee_normal + ef_normal]
            e_temp_pair_contact_vecs = [item for item in ee_contact_vec + ef_contact_vec]
            temp_pair_pnts.append(e_temp_pair_pnts)
            temp_pair_type.append(e_temp_pair_type)
            temp_pair_pnts_id.append(e_temp_pair_pnts_id)
            temp_pair_normals.append(e_temp_pair_normals)
            temp_pair_contact_vecs.append(e_temp_pair_contact_vecs)

        # surface → surface
        for i in self.surface_sampled_id[:]:
            detector_mesh = self.get_surface_detector(i, diameter=0.01)
            anchor_normal = self.pcd_normal[i]
            anchor = self.pcd_np[i]
            ff_pnts, ff_pnts_id, ff_normal, ff_contact_vec = self.find_antipodal(detector_mesh, anchor, i,
                                                                                 anchor_normal, self.surface_sampled,
                                                                                 self.surface_sampled_id,
                                                                                 self.surface_sampled_normal,
                                                                                 threshold=0.95)
            f_temp_pair_pnts = ff_pnts
            f_temp_pair_type = ['ff'] * len(ff_pnts)
            f_temp_pair_pnts_id = ff_pnts_id
            f_temp_pair_normals = ff_normal
            f_temp_pair_contact_vecs = ff_contact_vec
            temp_pair_pnts.append(f_temp_pair_pnts)
            temp_pair_type.append(f_temp_pair_type)
            temp_pair_pnts_id.append(f_temp_pair_pnts_id)
            temp_pair_normals.append(f_temp_pair_normals)
            temp_pair_contact_vecs.append(f_temp_pair_contact_vecs)

        self.pair_pnts = list(itertools.chain(*temp_pair_pnts))
        self.pair_type = list(itertools.chain(*temp_pair_type))
        self.pair_pnts_id = list(itertools.chain(*temp_pair_pnts_id))
        self.pair_normals = list(itertools.chain(*temp_pair_normals))
        self.pair_contact_vecs = list(itertools.chain(*temp_pair_contact_vecs))

    def find_antipodal(self, detector, anchor, anchor_id, anchor_normal, target_pnts, target_pnts_id, target_normal,
                       threshold=1.0):  # 判断两个点能不能成为对偶接触
        checker = trimesh.base.ray.ray_pyembree.RayMeshIntersector(detector)
        inside_points = []
        inside_points_id = []
        inside_normal = []
        anti_points = []
        anti_points_id = []
        anti_normal = []
        anti_contact_vectors = []
        try:
            inner_tf_list = checker.contains_points(target_pnts)
        except Exception as e:
            print(f"Exception occured: {e}")
            return [], [], [], []
        if len(inner_tf_list) == 0:
            print("No intersects found")
        for i, item in enumerate(inner_tf_list):
            if item:
                inside_points.append(target_pnts[i])
                inside_points_id.append(target_pnts_id[i])
                inside_normal.append(target_normal[i])
                contact_vector, pair_type = self.calculate_contact_vector(anchor_id, target_pnts_id[i], anchor,
                                                                          target_pnts[i], anchor_normal,
                                                                          target_normal[i])
                if pair_type == "have face" or pair_type == "only corner":
                    judge = contact_vector.dot(rm.unit_vector(target_pnts[i] - anchor)) >= threshold
                elif pair_type == "have edge":
                    judge = rm.unit_vector(target_normal[i]).dot(rm.unit_vector(anchor_normal)) <= -threshold
                else:
                    judge = True
                if judge:
                    anti_points.append([anchor, target_pnts[i]])
                    anti_points_id.append([anchor_id, target_pnts_id[i]])
                    anti_normal.append([anchor_normal, target_normal[i]])
                    anti_contact_vectors.append(contact_vector)
        return anti_points, anti_points_id, anti_normal, anti_contact_vectors

    def calculate_contact_vector(self, anchor_id, target_id, anchor, target, anchor_normal, target_normal):  # 计算接触方向
        anchor_type = self.get_point_type(anchor_id)
        target_type = self.get_point_type(target_id)
        if 'face' in (anchor_type, target_type):
            contact_vector = rm.unit_vector(target_normal if target_type == 'face' else -anchor_normal)
            pair_type = "have face"
        elif 'edge' in (anchor_type, target_type):
            contact_vector = rm.unit_vector(target - anchor)
            pair_type = "have edge"
        elif 'corner' in (anchor_type, target_type):
            contact_vector = rm.unit_vector(target - anchor)
            pair_type = "only corner"
        else:
            # unknown: fallback to vector between points
            contact_vector = rm.unit_vector(target - anchor)
            pair_type = "unknown"
        return contact_vector, pair_type

    def get_point_type(self, point_id):    # 判断点属于哪一类
        if point_id in list(self.surface_id):
            return 'face'
        elif point_id in list(self.vertex_id_clustered):
            return 'corner'
        elif point_id in list(self.edge_id):
            return 'edge'
        else:
            return 'unknown'

    def get_surface_detector(self, surface_id=None, diameter=0.01):   # 给面点生成检测器
        if surface_id is None:
            surface_id = self.surface_sampled_id[np.random.randint(0, len(self.surface_sampled_id) - 1)]
        surface_pnt = self.pcd_np[surface_id]
        surface_normal = -self.pcd_normal[surface_id]
        surface_normal_uniform = rm.unit_vector(surface_normal)
        pcd_scale = self.detect_scale
        homomat = rm.homomat_from_posrot(surface_pnt, rm.rotmat_from_normal(surface_normal_uniform))
        cyl = tpg.gen_cylinder(radius=diameter, height=pcd_scale, homomat=homomat)
        return cyl

    def get_edge_detector(self, edge_id=None, radius=0.01, threshold=0.008, maxIteration=100):  # 给边点生成检测器
        if not edge_id:
            edge_id = self.edge_sampled_id[np.random.randint(0, len(self.edge_sampled_id) - 1)]
        edge_pnt = self.pcd_np[edge_id]
        edge_normal = -self.pcd_normal[edge_id]
        edge_normal_uniform = rm.unit_vector(edge_normal)
        tangential_direction = self.fit_line_from_point(edge_pnt, radius, threshold, maxIteration)
        sec = tpg.gen_section(spos=edge_pnt, epos=edge_pnt + self.detect_scale * edge_normal_uniform,
                                height_vec=tangential_direction, height=0.02, angle=90, section=8)
        return sec

    def get_corner_detector(self, corner_id=None):  # 给角点生成检测器
        if corner_id is None:
            corner_id = self.corner_sampled_id[np.random.randint(0, len(self.corner_sampled_id) - 1)]
        corner_pnt = self.pcd_np[corner_id]
        corner_normal = -self.pcd_normal[corner_id]
        corner_normal_scale = np.linalg.norm(corner_normal)
        corner_normal_uniform = (1 / corner_normal_scale) * corner_normal
        cone = tpg.gen_cone(spos=corner_pnt + self.detect_scale * corner_normal_uniform, epos=corner_pnt,
                               radius=self.detect_scale, sections=24)
        return cone

    def fit_line_from_point(self, point, radius, threshold, maxIteration):   # 拟合边的方向
        if len(self.edge_pnt) == 0:
            return np.array([1.0, 0.0, 0.0])
        edge_pnt_pcd = pda.nparray_to_o3dpcd(self.edge_pnt)
        kdtree = o3d.geometry.KDTreeFlann(edge_pnt_pcd)
        [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        neighbor_points = self.edge_pnt[idx] if len(idx) > 0 else self.pcd_np[:2]
        if len(neighbor_points) < 2:
            [k, idx, _] = self.pcd_tree.search_radius_vector_3d(point, radius)
            neighbor_points = self.pcd_np[idx]
        line_model = pyrsc.Line()
        try:
            A, B, inliers = line_model.fit(neighbor_points, threshold, maxIteration)
            inliers_points = neighbor_points[inliers]
            inliers_mean = np.mean(inliers_points, axis=0)
            center_points = inliers_points - inliers_mean
            _, _, Vt = np.linalg.svd(center_points)
            inliers_direction = Vt[0]
        except Exception:
            pts = neighbor_points - neighbor_points.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts)
            inliers_direction = Vt[0]
        return inliers_direction

    def grasp_planning(self, rot_num=6, check_collision=False):  # 把接触点 → 变成抓取姿态
        grip_info_dict = {'jaw_width': [], 'jaw_center_pos': [], 'jaw_center_rotmat': [],
                          'hand_pos': [], 'hand_rotmat': []}
        for i, pair in enumerate(self.pair_pnts):
            finger_center = np.mean(pair, axis=0)
            jaw_width = np.linalg.norm(pair[1] - pair[0]) * 1.1
            if jaw_width <= 0:
                for key in grip_info_dict.keys():
                    grip_info_dict[key].append([])
                continue
            contact_vector = self.pair_contact_vecs[i] if i < len(self.pair_contact_vecs) else rm.unit_vector(pair[1] - pair[0])
            contact_rot = rm.rotmat_between_vectors(np.array([1, 0, 0]), contact_vector)
            rot_list = [rm.rotmat_from_axangle(contact_vector, angle).dot(contact_rot) for angle in
                        np.linspace(0, 2 * np.pi, rot_num)]
            final_rot_list = rot_list
            grip_info_dict['jaw_width'].append([jaw_width] * len(final_rot_list))
            grip_info_dict['jaw_center_pos'].append([finger_center] * len(final_rot_list))
            grip_info_dict['jaw_center_rotmat'].append(final_rot_list)
            grip_info_dict['hand_pos'].append([None] * len(final_rot_list))
            grip_info_dict['hand_rotmat'].append(final_rot_list)
        return grip_info_dict

    def get_pcd_bound(self):      # 计算点云边界
        if len(self.pcd_np) == 0:
            return np.zeros(3), np.zeros(3)
        min_bound = np.min(self.pcd_np, axis=0)
        max_bound = np.max(self.pcd_np, axis=0)
        return min_bound, max_bound

    def get_pcd_max_scale(self):   # 估计物体尺寸
        if len(self.pcd_np) == 0:
            return 1.0
        x_scale = np.min(self.pcd_np[:, 0]) - np.max(self.pcd_np[:, 0])
        y_scale = np.min(self.pcd_np[:, 1]) - np.max(self.pcd_np[:, 1])
        z_scale = np.min(self.pcd_np[:, 2]) - np.max(self.pcd_np[:, 2])
        max_scale = np.max(np.abs([x_scale, y_scale, z_scale]))
        return max_scale if max_scale != 0 else 1.0
