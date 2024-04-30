import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# object
object = cm.CollisionModel("objects/box_text.stl")
object.set_rgba([.9, .75, .35, 1])
object.attach_to(base)

# hnd_s
gripper_s = rf.reconfgripper().lft
gripper_m = rf.reconfgripper()
grasp_info_list = gpa.plan_grasps(gripper_s, object,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('holder', grasp_info_list, './', 'cobg_holder_grasps.pickle')
# grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')

a=gripper_m.pos
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=[0, 1, 0, 1]).attach_to(base)

    m_pos = np.array([-.0626, 0, -.1373]) + gripper_s.rotmat.dot(a)
    m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
    gripper_m.fix_to(m_pos, hnd_rotmat)
    gripper_m.mg_open()
    gripper_m.gen_meshmodel(rgba=[0, 0, 1, .3]).attach_to(base)
base.run()
