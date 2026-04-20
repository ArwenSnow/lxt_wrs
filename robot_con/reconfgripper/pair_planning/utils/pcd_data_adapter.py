import numpy as np
import open3d as o3d


def nparray_to_o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


def o3dpcd_to_parray(o3dpcd, return_normals=False):
    if return_normals:
        if o3dpcd.has_normals():
            return [np.asarray(o3dpcd.points), np.asarray(o3dpcd.normals)]
        else:
            o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
            return [np.asarray(o3dpcd.points), np.asarray(o3dpcd.normals)]
    else:
        return np.asarray(o3dpcd.points)
