import numpy as np
import utils.robot_math as rm
from scipy.spatial import ConvexHull


class StabilityAnalyzer:
    def __init__(self, obj_name, obj_path, base=None, contact_u=0.25, max_deform_dist=.003,
                 debug=False, benchmark=False, silent=False, pair_data=None, pcd_np=None):
        self.debug = False
        self.contact_u = contact_u
        self.max_deform_dist = max_deform_dist
        self.obj_name = obj_name
        self.obj_path = obj_path
        self.base = base
        self.benchmark = benchmark
        self.silent = silent or benchmark

        # pcd in memory (np array Nx3)
        self.pcd_np = np.asarray(pcd_np) if pcd_np is not None else np.asarray([])

        # pair data either provided or empty
        if pair_data is not None:
            self.pair_pnts = pair_data.get('pair_pnts', [])
            self.pair_contact_vecs = pair_data.get('pair_contact_vecs', [])
        else:
            self.pair_pnts = []
            self.pair_contact_vecs = []

        self.obj_com = np.array([0.0, 0.0, 0.0])

    def filter_points_by_finger_model(self, point, direction, contact_region=.01, distance_thresh=.003):
        direction = rm.unit_vector(direction)
        if self.pcd_np is None or len(self.pcd_np) == 0:
            return np.asarray([])
        vecs = self.pcd_np - point
        distance_1 = np.linalg.norm(vecs, axis=1)
        proj = np.dot(vecs, direction)
        mask = (distance_1 < contact_region) & (proj > 0) & (proj < distance_thresh)
        return self.pcd_np[mask]

    def gen_FC_vec(self, coefficient, height=1, polygon=6, normal=np.array([0, 0, 1])):
        init_mat = np.zeros([polygon, 3])
        init_vec = np.array([0, coefficient, 1]) * height
        tf_rot = rm.rotmat_between_vectors(np.array([0, 0, 1]), normal)
        init_mat[0, :] = rm.homomat_transform_points(rm.homomat_from_posrot([0, 0, 0], tf_rot), init_vec)
        angle = np.radians(360.0) / polygon
        for i in range(1, polygon):
            rotMat = rm.rotmat_from_axangle([0, 0, 1], angle * i)
            init_mat[i, :] = rm.homomat_transform_points(rm.homomat_from_posrot([0, 0, 0], tf_rot),
                                                         np.dot(rotMat, init_vec))
        return init_mat

    def gen_FC(self, position, u=0.25, normal=np.array([0, 0, 1]), polygon=6, height=1, show=False):
        return self.gen_FC_vec(coefficient=u, height=height, polygon=polygon, normal=normal)

    def minidistance_hull(self, p, hull):
        if not isinstance(hull, ConvexHull):
            hull = ConvexHull(hull, qhull_options="QJ")
        return np.max(np.dot(hull.equations[:, :-1], p.T).T + hull.equations[:, -1], axis=-1)

    def cal_stability(self, point_clutter_1, point_center_1, point_clutter_2, point_center_2, obj_com,
                      applied_force_1, applied_force_2, mass_gravity, contact_u=0.25):
        if len(point_clutter_1) < 3 or len(point_clutter_2) < 3:
            return -1
        ch1 = ConvexHull(point_clutter_1, qhull_options='QJ')
        ch2 = ConvexHull(point_clutter_2, qhull_options='QJ')
        pc_sparse_1 = point_clutter_1[ch1.vertices]
        pc_sparse_2 = point_clutter_2[ch2.vertices]
        applied_force_1_weight = np.linalg.norm(applied_force_1) / len(pc_sparse_1)
        applied_force_2_weight = np.linalg.norm(applied_force_2) / len(pc_sparse_2)
        FC_list_1 = []
        FC_list_2 = []
        for i, p in enumerate(pc_sparse_1):
            FC_list_1.append(
                self.gen_FC(position=p, u=contact_u, height=applied_force_1_weight, normal=applied_force_1))
        for i, p in enumerate(pc_sparse_2):
            FC_list_2.append(
                self.gen_FC(position=p, u=contact_u, height=applied_force_2_weight, normal=applied_force_2))
        wrench_list_1 = []
        wrench_list_2 = []
        for i, force in enumerate(FC_list_1):
            for f in force:
                wrench_list_1.append(np.hstack([f, np.cross(pc_sparse_1[i] - obj_com, f)]))
        for i, force in enumerate(FC_list_2):
            for f in force:
                wrench_list_2.append(np.hstack([f, np.cross(pc_sparse_2[i] - obj_com, f)]))
        wrench_sets = np.asarray(wrench_list_1 + wrench_list_2)
        wrench_sets = np.unique(wrench_sets, axis=0)
        if wrench_sets.shape[0] < 4:
            return -1
        hull = ConvexHull(wrench_sets, qhull_options="QJ Pp")
        mass_wrench = np.zeros(6)
        mass_wrench[:3] = mass_gravity
        mindistance = self.minidistance_hull(np.zeros(6) - mass_wrench, hull)
        if mindistance > 0:
            return 0
        return abs(mindistance)

    def analyze_stability(self, debug_id=0, c1_force_mag=2.0, c2_force_mag=2.0,
                          mass_gravity=np.array([0, 0, -9.81]), transformation=np.eye(4), save=False):
        pair_stability = []
        pair_c1_pnts = []
        pair_c2_pnts = []
        for i, contact_pair_pnts in enumerate(self.pair_pnts[debug_id:]):
            try:
                p1, p2 = contact_pair_pnts
            except Exception:
                pair_stability.append(-1)
                continue
            d1 = rm.unit_vector(self.pair_contact_vecs[debug_id:][i]) if len(
                self.pair_contact_vecs) > 0 else rm.unit_vector(p2 - p1)
            d2 = -d1
            force_c1 = d1 * c1_force_mag
            force_c2 = d2 * c2_force_mag
            c1_points = self.filter_points_by_finger_model(point=p1, direction=d1)
            c2_points = self.filter_points_by_finger_model(point=p2, direction=d2)
            pair_c1_pnts.append(c1_points)
            pair_c2_pnts.append(c2_points)
            if len(c1_points) == 0 or len(c2_points) == 0:
                pair_stability.append(-1)
            else:
                stability = self.cal_stability(point_clutter_1=c1_points,
                                               point_center_1=p1,
                                               point_clutter_2=c2_points,
                                               point_center_2=p2,
                                               obj_com=self.obj_com,
                                               applied_force_1=force_c1,
                                               applied_force_2=force_c2,
                                               mass_gravity=mass_gravity,
                                               contact_u=self.contact_u)
                pair_stability.append(stability)
        return pair_stability
