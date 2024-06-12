import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.ag145.ag145 as ag145
import numpy as np

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# object
object = cm.CollisionModel("objects/box_text.stl")
object.set_rgba([.9, .75, .35, 1])
object.attach_to(base)

# hnd_s
gripper_s = ag145.Ag145()

grasp_info_list = gpa.plan_grasps(gripper_s, object,
                                  angle_between_contact_normals=math.radians(177),
                                  openning_direction='loc_y',
                                  max_samples=3, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.005)
gpa.write_pickle_file('box', grasp_info_list, './', 'ag145_grasps.pickle')

a = gripper_s.pos
b = gripper_s.rotmat

#an example
grasp_info = grasp_info_list[0]
jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
cm.gen_sphere(pos=jaw_center_pos, radius=0.002, rgba=[1, 0, 0, 1]).attach_to(base)
gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
gripper_s.gen_meshmodel(rgba=[1, 0, 0, .1]).attach_to(base)



# for grasp_info in grasp_info_list:
#     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
#     cm.gen_sphere(pos=hnd_pos, radius=0.002, rgba=[1, 0, 0, 1]).attach_to(base)
#     gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
#     m_pos = hnd_pos - np.dot(hnd_pos,hnd_rotmat)
#     hnd_rotmat = hnd_rotmat
#     gripper_s.fix_to(m_pos, hnd_rotmat)
#     gripper_s.gen_meshmodel(rgba=[0, 0, 1, .1]).attach_to(base)
#     m_pos = hnd_pos
#     hnd_rotmat = hnd_rotmat
#     gripper_s.fix_to(m_pos, hnd_rotmat)
#     gripper_s.gen_meshmodel(rgba=[1, 0, 0, .1]).attach_to(base)
base.run()