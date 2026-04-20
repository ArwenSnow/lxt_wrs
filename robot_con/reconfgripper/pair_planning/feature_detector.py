# 完全无可视化、无文件保存的 FeatureDetector（返回内存数据字典）
from sklearn.cluster import DBSCAN
import numpy as np
import open3d as o3d
import utils.pcd_data_adapter as pda
import time


class FeatureDetector:

    def __init__(self, objpath, objname, base=None):
        self.objname = objname
        self.objpath = objpath

    def main_detector(self, r_1=0.9, r_2=0.7, k=30, eps=0.005, remove_concave=True):
        # All I/O flags ignored: this class never writes files or visualizes.
        self._get_pcd(self.objpath, sample="poisson", pcd_num=10000, neighbour=50, radius=0.05)
        self.pcd_sample, self.pcd_sample_np, self.pcd_normal_sample, self.pcd_normal_sample_np = self._sample_contact()

        t_edge_start = time.time()
        self.detect_edge(threshold=r_1, r=0.005, k=k, check_concave=True)
        t_edge_end = time.time()

        t_vertex_start = time.time()
        self.detect_vertex(threshold=r_2, r=0.01, k=int(k * 0.5))
        t_vertex_end = time.time()

        self.cluster_vertex_result(eps=eps, min_samples=1, remove_concave=remove_concave)

        # 返回内存字典，供 ContactDetector 直接使用
        features = {
            'pcd': getattr(self, 'pcd', None),
            'pcd_np': getattr(self, 'pcd_np', None),
            'pcd_id': getattr(self, 'ind', None),
            'pcd_normal': getattr(self, 'pcd_normal', None),
            'pcd_sample': getattr(self, 'pcd_sample', None),
            'pcd_sample_np': getattr(self, 'pcd_sample_np', None),
            'pcd_normal_sample': getattr(self, 'pcd_normal_sample', None),
            'surface_pnt': getattr(self, 'surface_pnt', None),
            'surface_normal': getattr(self, 'surface_normal', None),
            'surface_id': getattr(self, 'surface_id', None),
            'edge_pnt': getattr(self, 'edge_pnt', None),
            'edge_normal': getattr(self, 'edge_normal', None),
            'edge_id': getattr(self, 'edge_id', None),
            'vertex_pnt': getattr(self, 'vertex_pnt', None),
            'vertex_normal': getattr(self, 'vertex_normal', None),
            'vertex_id': getattr(self, 'vertex_id', None),
            'vertex_pnt_clustered': getattr(self, 'vertex_pnt_clustered', None),
            'vertex_normal_clustered': getattr(self, 'vertex_normal_clustered', None),
            'vertex_id_clustered': getattr(self, 'vertex_id_clustered', None),
            'concave_edge_id': getattr(self, 'concave_edge_id', None),
            'edge_detect_parameter': getattr(self, 'edge_detect_parameter', None),
            'vertex_detect_parameter': getattr(self, 'vertex_detect_parameter', None),
            'vertex_clustered_parameter': getattr(self, 'vertex_clustered_parameter', None),
            'timings': {
                'edge_detection': t_edge_end - t_edge_start,
                'vertex_detection': t_vertex_end - t_vertex_start
            }
        }
        print("Detected features:")
        print(f"  Surface points: {len(features['surface_pnt'])}")
        print(f"  Edge points: {len(features['edge_pnt'])}")
        print(f"  Vertex points (clustered): {len(features['vertex_pnt_clustered'])}")
        return features

    def _get_pcd(self, objpath, sample="poisson", pcd_num=10000, neighbour=50, radius=0.05):
        mesh = o3d.io.read_triangle_mesh(objpath)
        mesh.compute_vertex_normals()
        if sample == "poisson":
            pcd = mesh.sample_points_poisson_disk(number_of_points=pcd_num)
        elif sample == "uniform":
            pcd = mesh.sample_points_uniformly(number_of_points=pcd_num, use_triangle_normal=False)
        else:
            pcd = mesh.sample_points_poisson_disk(number_of_points=pcd_num)
        self.pcd, self.ind = pcd.remove_radius_outlier(nb_points=neighbour, radius=radius)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.pcd_np = pda.o3dpcd_to_parray(self.pcd)
        # pcd normals initialized empty; later populated for features
        self.pcd_normal = []

    def detect_edge(self, threshold=0.5, r=0.003, k=20, check_concave=True):
        # produce in-memory lists for edge/surface and pcd_normal
        self.edge_pnt = []
        self.edge_normal = []
        self.edge_id = []
        self.concave_edge_id = []
        self.surface_pnt = []
        self.surface_normal = []
        self.surface_id = []
        self.edge_detect_parameter = {'threshold': threshold, 'r': r, 'k': k}

        for i, anchor in enumerate(self.pcd_sample_np):
            anchor, anchor_normal, pcd_neighbor_np, pcd_neighbor_normal_np, is_surface = (
                self._get_neibour_detect(anchor, i, threshold=threshold, radius=r, k=k))
            if check_concave and (not is_surface):
                if self._is_concave(anchor, anchor_normal, pcd_neighbor_np):
                    self.concave_edge_id.append(i)

        self.surface_pnt = np.asarray(self.surface_pnt)
        self.surface_normal = np.asarray(self.surface_normal)
        self.surface_id = np.asarray(self.surface_id)
        self.edge_pnt = np.asarray(self.edge_pnt)
        self.edge_normal = np.asarray(self.edge_normal)
        self.edge_id = np.asarray(self.edge_id)
        self.pcd_normal = np.asarray(self.pcd_normal)

    def detect_vertex(self, threshold=1.0, r=0.01, k=20):
        self.vertex_pnt = []
        self.vertex_normal = []
        self.vertex_id = []
        self.vertex_detect_parameter = {'threshold': threshold, 'r': r, 'k': k}
        if len(self.edge_pnt) == 0:
            self.vertex_pnt = np.asarray([])
            self.vertex_normal = np.asarray([])
            self.vertex_id = np.asarray([])
            return

        self.edge_pcd_tree = o3d.geometry.KDTreeFlann(pda.nparray_to_o3dpcd(np.asarray(self.edge_pnt)))
        for id_num, pnt in enumerate(self.edge_pnt):
            _, idx, _ = self.edge_pcd_tree.search_knn_vector_3d(pnt, k)
            pcd_neighbor_normal_np = np.vstack([self.edge_normal[idx[1:]], self.edge_normal[id_num]])
            pcd_neighbor_np = np.vstack([self.edge_pnt[idx[1:]], pnt])
            pcd_neighbor_np, pcd_neighbor_normal_np = self.cluster_input(pnt, pcd_neighbor_np,
                                                                         pcd_neighbor_normal_np,
                                                                         normal_cluster=True)
            if len(pcd_neighbor_np) == 1:
                continue
            anchor_normal = self.edge_normal[id_num]
            if self._is_vertex(pcd_neighbor_np, threshold):
                self.vertex_pnt.append(pnt)
                self.vertex_normal.append(anchor_normal)
                self.vertex_id.append(self.edge_id[id_num])

        self.vertex_pnt = np.asarray(self.vertex_pnt)
        self.vertex_normal = np.asarray(self.vertex_normal)
        self.vertex_id = np.asarray(self.vertex_id)

    def cluster_vertex_result(self, eps, min_samples, remove_concave=True):
        self.vertex_clustered_parameter = {'eps': eps, 'min_samples': min_samples}
        if len(self.vertex_pnt) == 0:
            self.vertex_pnt_clustered = np.asarray([])
            self.vertex_normal_clustered = np.asarray([])
            self.vertex_id_clustered = np.asarray([])
            return
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.vertex_pnt)
        unique_labels = set(labels)
        vertex_id_clustered = []
        for label in unique_labels:
            cluster_points_id = np.where(labels == label)[0]
            cluster_points = self.vertex_pnt[cluster_points_id]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_distance = np.linalg.norm(cluster_points - cluster_mean, axis=1)
            min_distance_index = np.argmin(cluster_distance)
            corner_clustered_id = cluster_points_id[min_distance_index]
            vertex_id_clustered.append(self.vertex_id[corner_clustered_id])
        if remove_concave:
            remove_id = []
            for cluster_id in vertex_id_clustered:
                cluster_pnt = self.pcd_np[cluster_id]
                cluster_pnt_normal = self.pcd_normal[cluster_id]
                _, idx, _ = self.pcd_tree.search_radius_vector_3d(cluster_pnt, eps)
                cluster_pnt_neighbor = self.pcd_np[idx]
                if self._is_concave(cluster_pnt, cluster_pnt_normal, cluster_pnt_neighbor):
                    remove_id.append(cluster_id)
                if len(np.intersect1d(self.concave_edge_id, idx)) != 0:
                    remove_id.append(cluster_id)
            vertex_id_clustered = list(set(vertex_id_clustered) - set(remove_id))

        self.vertex_pnt_clustered = np.asarray(self.pcd_np[vertex_id_clustered])
        self.vertex_id_clustered = np.asarray(vertex_id_clustered)
        self.vertex_normal_clustered = np.asarray(self.pcd_normal[vertex_id_clustered])

    def _sample_contact(self, everynum=3):
        pcd_sample = self.pcd
        pcd_normal_sample = self.pcd.normals
        pcd_normal_sample_np = np.asarray(pcd_normal_sample)
        pcd_sample_np = pda.o3dpcd_to_parray(pcd_sample)
        return pcd_sample, pcd_sample_np, pcd_normal_sample, pcd_normal_sample_np

    def _get_neibour_detect(self, anchor, id_num, threshold, radius=0.0025, k=20, toggle_nb=False):
        _, idx, _ = self.pcd_tree.search_knn_vector_3d(anchor, k)
        pcd_neighbor_normal_np_ori = np.vstack([self.pcd_normal_sample_np[idx[1:]], self.pcd_normal_sample_np[id_num]])
        pcd_neighbor_np = np.vstack([self.pcd_np[idx[1:]], anchor])
        pcd_neighbor_np, pcd_neighbor_normal_np = self.cluster_input(anchor, pcd_neighbor_np,
                                                                     pcd_neighbor_normal_np_ori,
                                                                     normal_cluster=True)
        if len(pcd_neighbor_np) == 1:
            pass
        is_surface = self._is_surface(anchor, pcd_neighbor_np, threshold)
        if is_surface:
            _, surface_normal_idx, _ = self.pcd_tree.search_knn_vector_3d(anchor, int(k / 2))
            surface_normal = self.pcd_normal_sample_np[surface_normal_idx[1:]]
            anchor_normal = surface_normal.mean(axis=0)
            self.surface_pnt.append(anchor)
            self.surface_normal.append(anchor_normal)
            self.surface_id.append(id_num)
        else:
            anchor_normal = pcd_neighbor_normal_np_ori.mean(axis=0)
            self.edge_pnt.append(anchor)
            self.edge_normal.append(anchor_normal)
            self.edge_id.append(id_num)
        self.pcd_normal.append(anchor_normal)
        return anchor, anchor_normal, pcd_neighbor_np, pcd_neighbor_normal_np, is_surface

    def _is_surface(self, anchor, pcd_neighbor_np, threshold):
        if len(pcd_neighbor_np) == 1:
            return True
        datamean = pcd_neighbor_np.mean(axis=0)
        distance_neighbor_center = [np.linalg.norm(anchor - p) for p in pcd_neighbor_np]
        distance_neighbor_center_min = np.sort(np.array(distance_neighbor_center))[1]
        distance_query = np.linalg.norm(datamean - anchor)
        return not (distance_query > (distance_neighbor_center_min * threshold))

    def _is_concave(self, anchor, normal, neighbors):
        center_point = neighbors.mean(axis=0)
        vector = center_point - anchor
        anchor_normal = anchor + normal
        norm_vector = np.linalg.norm(vector) + 1e-12
        norm_anchor_normal = np.linalg.norm(anchor_normal) + 1e-12
        cos_theta = np.clip(vector.dot(anchor_normal) / (norm_vector * norm_anchor_normal), -1.0, 1.0)
        angle = np.arccos(cos_theta) * 180 / np.pi
        return angle < 90

    def _is_vertex(self, pcd_neighbor_np, threshold):
        datamean = pcd_neighbor_np.mean(axis=0)
        eigen_vals, eigen_vecs = self.pca(pcd_neighbor_np - datamean)
        PC = eigen_vals / np.sum(eigen_vals, axis=0)
        return np.max(PC) < threshold

    def pca(self, X):
        n, m = X.shape
        if n <= 1:
            # degenerate
            return np.array([1.0, 0.0, 0.0]), np.eye(3)
        Xc = X - X.mean(axis=0)
        eigen_vals, eigen_vecs = np.linalg.eig(np.dot(Xc.T, Xc) / (n - 1))
        # sort descending
        order = np.argsort(-eigen_vals)
        return eigen_vals[order], eigen_vecs[:, order]

    def cluster_input(self, anchor, points, normals, normal_cluster=False):
        points_scale = self.scale_compute(points)
        dbscan_points = DBSCAN(eps=points_scale / 2, min_samples=1)
        points_labels = dbscan_points.fit_predict(points)
        unique_labels = set(points_labels)
        unique_labels.discard(-1)
        if len(unique_labels) != 1:
            target_label = points_labels[np.where((points == anchor).all(axis=1))[0][0]]
            result_points = points[points_labels == target_label]
            result_normals = normals[points_labels == target_label]
        else:
            result_points = points
            result_normals = normals
        if normal_cluster:
            dbscan_normals = DBSCAN(eps=0.5, min_samples=1)
            normals_labels = dbscan_normals.fit_predict(result_normals)
            unique_normals_labels = set(normals_labels)
            if len(unique_normals_labels) != 1:
                target_label = normals_labels[np.where((result_points == anchor).all(axis=1))[0][0]]
                result_points = result_points[normals_labels == target_label]
                result_normals = result_normals[normals_labels == target_label]
        return result_points, result_normals

    def scale_compute(self, data):
        if data is None or len(data) == 0:
            return 1.0
        x_scale = np.min(data[:, 0]) - np.max(data[:, 0])
        y_scale = np.min(data[:, 1]) - np.max(data[:, 1])
        z_scale = np.min(data[:, 2]) - np.max(data[:, 2])
        max_scale = np.max(np.abs([x_scale, y_scale, z_scale]))
        if max_scale == 0 or np.isnan(max_scale):
            max_scale = 1
        return max_scale


if __name__ == '__main__':
    obj_name = 'tetrahedron'
    obj_path = f'./test_obj/{obj_name}.stl'
    fd = FeatureDetector(objpath=obj_path, objname=obj_name)
    features = fd.main_detector(r_1=0.9, r_2=0.7, k=30, eps=0.005, remove_concave=True)
