#xc330gripper grasps the object directly
import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1_old.xc330gripper1 as xc

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# object
object = cm.CollisionModel("objects/box_text.stl")
object.set_rgba([.9, .75, .35, 1])
object.attach_to(base)

# hnd_s
gripper_s = xc.xc330gripper()
grasp_info_list = gpa.plan_grasps(gripper_s, object,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('holder', grasp_info_list, './', 'cobg_holder_grasps.pickle')
# grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')

# grasp_info = grasp_info_list[0]
# jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
# gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
# gripper_s.gen_meshmodel(rgba=[0, 1, 0, 1]).attach_to(base)

for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
base.run()