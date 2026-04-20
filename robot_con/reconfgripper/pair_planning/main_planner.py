# headless 全内存流水线：FeatureDetector -> ContactDetector -> StabilityAnalyzer
from feature_detector import FeatureDetector
from contact_detector import ContactDetector
from grasp_stability import StabilityAnalyzer
import time
import numpy as np
import modeling.collision_model as cm


def benchmark_pipeline(object_path, object_name):
    # Feature detection
    t0 = time.perf_counter()
    fd = FeatureDetector(objpath=object_path, objname=object_name)
    features = fd.main_detector(r_1=0.9, r_2=0.7, k=30, eps=0.005, remove_concave=True)
    t1 = time.perf_counter()

    # Contact detection (in-memory features)
    cd = ContactDetector(objpath=object_path, objname=object_name, feature_data=features)
    t2_start = time.perf_counter()
    cd_result = cd.main_detector(voxel_size=0.04)
    t2 = time.perf_counter()

    # Stability analysis (in-memory pairs and pcd)
    pair_data = {'pair_pnts': cd_result.get('pair_pnts', []),
                 'pair_contact_vecs': cd_result.get('pair_contact_vecs', [])}
    sa = StabilityAnalyzer(obj_name=object_name, obj_path=object_path, pair_data=pair_data, pcd_np=features.get('pcd_np'))
    t3_start = time.perf_counter()
    stability = sa.analyze_stability(c1_force_mag=100.0, c2_force_mag=100.0,
                                     mass_gravity=np.array([0, 0, 0.01]))

    t3 = time.perf_counter()
    print(f"Stable contact pairs: {len(np.where(np.array(stability) > 0)[0])}")

    fd_time = t1 - t0
    cd_time = t2 - t2_start
    sa_time = t3 - t3_start
    total_time = fd_time + cd_time + sa_time
    print(f"FeatureDetector_time:{fd_time:.6f}s")
    print(f"ContactDetector_time:{cd_time:.6f}s")
    print(f"StabilityAnalyzer_time:{sa_time:.6f}s")
    print(f"Total_time:{total_time:.6f}s")

    return ({'feature_time': fd_time, 'contact_time': cd_time, 'stability_time': sa_time, 'total_time': total_time},
            cd_result, stability)


if __name__ == "__main__":
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import robot_sim.robots.Franka_research3.Franka_research3 as Fr
    import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rg

    base = wd.World(cam_pos=[-2, 4, 1.5], lookat_pos=[0, 0, 0])
    obj_name = 'tetrahedron'
    obj_path = f'./test_obj/{obj_name}.stl'

    _, contact_result, stability_result = benchmark_pipeline(obj_path, obj_name)
    valid = [(i, v) for i, v in enumerate(stability_result) if v >= 0]
    min_index = min(valid, key=lambda x: x[1])[0] if valid else None

    grasp_info = contact_result.get('grip_info')
    jaw_center_pos = grasp_info.get('jaw_center_pos')[min_index]
    jaw_center_rotmat = grasp_info.get('jaw_center_rotmat')[min_index]
    jaw_width = grasp_info.get('jaw_width')[min_index]

    # ========================object===================================
    obj = cm.CollisionModel(obj_path)
    obj.attach_to(base)
    robot = Fr.Franka_research3()
    mg = rg.Reconfgripper()
    lg = rg.Reconfgripper().lft
    rg = rg.Reconfgripper().rgt
    lg.grip_at_with_jcpose(jaw_center_pos[0], jaw_center_rotmat[0], 0.028)
    lg.gen_meshmodel().attach_to(base)

    base.run()


