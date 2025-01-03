import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# finger_type
finger_type = 'b'

# hnd_type
g = 'l'

# finger_s
if finger_type == 'a':
    finger = "objects/finger_a.stl"
elif finger_type == 'b':
    finger = "objects/finger_b.stl"
elif finger_type == 'c':
    finger = "objects/finger_c.stl"
else:
    raise ValueError(f"Invalid finger type selected: {finger_type}")

finger_1 = cm.CollisionModel(finger)
finger_1.set_rgba([.7, .7, .7, 1])
finger_2 = cm.CollisionModel(finger)
finger_2.set_rgba([.7, .7, .7, 1])


# hnd_s
if g == 'l':
    gripper = rf.reconfgripper().lft
elif g == 'r':
    gripper = rf.reconfgripper().rgt
gripper_m = rf.reconfgripper()
gripper_b = rf.reconfgripper().body

# tool for planning finger grasps
tool_a = cm.CollisionModel("objects/tool_a.stl")
tool_b = cm.CollisionModel("objects/tool_b_c.stl")
tool_c = cm.CollisionModel("objects/tool_b_c.stl")

# object_s
object_1 = cm.CollisionModel("objects/screw.stl")
object_1.set_rgba([.9, .75, .35, 1])
object_2 = cm.CollisionModel("objects/bigbox.stl")
object_2.set_rgba([.9, .75, .35, 1])
object_3 = cm.CollisionModel("objects/cylinder.stl")
object_3.set_rgba([.9, .75, .35, 1])

if finger_type == 'a':
    tool = tool_a
    target_object = object_1
    target_object.attach_to(base)
elif finger_type == 'b':
    tool = tool_b
    target_object = object_2
    target_object.attach_to(base)
elif finger_type == 'c':
    tool = tool_c
    target_object = object_3
    target_object.attach_to(base)

# gripper_grasp_finger
grasp_info_list = gpa.plan_grasps(gripper, tool,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=2, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('holder', grasp_info_list, './', 'cobg_holder_grasps.pickle')
# grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')

finger_grasp_info_list = []
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    x, y, z = jaw_center_pos
    x_axis_direction = jaw_center_rotmat[:, 0]
    desired_direction = np.array([-1, 0, 0])
    cos_angle = np.dot(x_axis_direction, desired_direction) / (
            np.linalg.norm(x_axis_direction) * np.linalg.norm(desired_direction))
    if (g == 'l' and cos_angle == -1) or (g == 'r' and cos_angle == 1):
        finger_grasp_info_list.append(grasp_info)

# grasp_info = finger_grasp_info_list[0]
# jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
#
# # update jawwidth_rng and jaw_center_pos
# gripper_b.set_finger_type(finger_type, jaw_center_pos, jaw_center_rotmat, g)
# m = gripper_b.jaw_center_pos
# n = gripper_b.jawwidth_rng
#
# # finger_grasp_object
# object_grasp_info_list = gpa.plan_grasps(gripper_b, target_object,
#                                   angle_between_contact_normals=math.radians(160),
#                                   openning_direction='loc_x',
#                                   max_samples=2, min_dist_between_sampled_contact_points=.005,
#                                   contact_offset=.001)
#
# new_object_grasp_info_list = []
# for object_grasp_info in object_grasp_info_list:
#     m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info
#     x, y, z = m_jaw_center_pos
#     if finger_type == 'a':
#         new_object_grasp_info_list.append(object_grasp_info)
#     if finger_type == 'b':
#         new_object_grasp_info_list.append(object_grasp_info)
#     if finger_type == 'c':
#         if z <= -1e-08 or z >= 1e-08:
#             new_object_grasp_info_list.append(object_grasp_info)
#
# new_object_grasp_info = new_object_grasp_info_list[0]
# m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = new_object_grasp_info
#
# finger_1_new = cm.CollisionModel(finger)
# finger_1_new.set_rgba([.7, .7, .7, 1])
# finger_2_new = cm.CollisionModel(finger)
# finger_2_new.set_rgba([.7, .7, .7, 1])
#
# if finger_type == 'a':
#     m_jaw_width = m_jaw_width - .00287
# elif finger_type == 'b':
#     m_jaw_width = m_jaw_width - .237
# elif finger_type == 'c':
#     m_jaw_width = m_jaw_width - .03959
#
# if g == 'l':
#     gripper_m.lg_jaw_to(jaw_width)
#     m_rotmat = hnd_rotmat
#     m_pos = np.array([-.053 - m_jaw_width / 2, .018, -.135])
#     m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
# elif g =='r':
#     gripper_m.rg_jaw_to(jaw_width)
#     m_rotmat = hnd_rotmat
#     m_pos = np.array([.053 + m_jaw_width / 2, -.018, -.135])
#     m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
#
# gripper_m.fix_to(m_pos, m_rotmat)
# gripper_m.mg_jaw_to(m_jaw_width)
# # gripper_m.gen_meshmodel().attach_to(base)
#
# if g == 'l':
#     gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="l")
#     finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
#     finger_2.set_pos(finger_2_pos)
#     finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#     finger_2.set_rotmat(finger_2_rotmat)
#     # finger_1.attach_to(base)
#     # finger_2.attach_to(base)
# elif g =='r':
#     gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="r")
#     finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
#     finger_2.set_pos(finger_2_pos)
#     finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#     finger_2.set_rotmat(finger_2_rotmat)
#     # finger_1.attach_to(base)
#     # finger_2.attach_to(base)
# gripper_m.gripper_bool(g='l')
# gripper_m.gripper_bool(g='r')
# gripper_m.gripper_bool(g='m')
#
# gripper_b.grip_at_with_jcpose(m_jaw_center_pos, m_jaw_center_rotmat, m_jaw_width)
# a = gripper_b.pos
# b = gripper_b.rotmat
# gripper_m.fix_to(a, b)
# gripper_m.mg_jaw_to(m_jaw_width)
#
# c = np.dot(b, m_rotmat.T)
# finger_1_pos_new = np.dot(c, np.array([0, 0, 0]) - m_pos) + a
# finger_1_rotmat_new = c
# finger_1_new.set_pos(finger_1_pos_new)
# finger_1_new.set_rotmat(finger_1_rotmat_new)
# objcm_1 = cm.CollisionModel(finger_2_new, cdprimit_type='box', expand_radius=.1)
#
# finger_2_pos_new = np.dot(c, finger_2_pos - m_pos) + a
# finger_2_rotmat_new = np.dot(c, finger_2_rotmat)
# finger_2_new.set_pos(finger_2_pos_new)
# finger_2_new.set_rotmat(finger_2_rotmat_new)
# objcm_2 = cm.CollisionModel(finger_2_new, cdprimit_type='box', expand_radius=.1)
#
# if not target_object.is_mcdwith([finger_2_new, finger_1_new]):
#     finger_1_new.attach_to(base)
#     finger_2_new.attach_to(base)
#     gripper_m.gen_meshmodel().attach_to(base)

for grasp_info in finger_grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info

    # update jawwidth_rng and jaw_center_pos
    gripper_b.set_finger_type(finger_type, jaw_center_pos, jaw_center_rotmat, g)
    m = gripper_b.jaw_center_pos
    n = gripper_b.jawwidth_rng

    # finger_grasp_object
    object_grasp_info_list = gpa.plan_grasps(gripper_b, target_object,
                                             angle_between_contact_normals=math.radians(160),
                                             openning_direction='loc_x',
                                             max_samples=2, min_dist_between_sampled_contact_points=.005,
                                             contact_offset=.001)

    # object_grasp_info = object_grasp_info_list[0]
    # m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info

    new_object_grasp_info_list = []
    for object_grasp_info in object_grasp_info_list:
        m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info
        x, y, z = m_jaw_center_pos
        if finger_type == 'a':
            new_object_grasp_info_list.append(object_grasp_info)
        if finger_type == 'b':
            new_object_grasp_info_list.append(object_grasp_info)
        if finger_type == 'c':
            if z <= -1e-08 or z >= 1e-08:
                new_object_grasp_info_list.append(object_grasp_info)

    for object_grasp_info in object_grasp_info_list:
        m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info

        new_object_grasp_info_list = []
        for object_grasp_info in object_grasp_info_list:
            m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info
            x, y, z = m_jaw_center_pos
            if finger_type == 'a':
                new_object_grasp_info_list.append(object_grasp_info)
            if finger_type == 'b':
                new_object_grasp_info_list.append(object_grasp_info)
            if finger_type == 'c':
                if z <= -1e-08 or z >= 1e-08:
                    new_object_grasp_info_list.append(object_grasp_info)

        for new_object_grasp_info in new_object_grasp_info_list:
            m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = new_object_grasp_info

            finger_1_new = cm.CollisionModel(finger)
            finger_1_new.set_rgba([.7, .7, .7, 1])
            finger_2_new = cm.CollisionModel(finger)
            finger_2_new.set_rgba([.7, .7, .7, 1])

            if finger_type == 'a':
                m_jaw_width = m_jaw_width - .00287
            elif finger_type == 'b':
                m_jaw_width = m_jaw_width - .237
            elif finger_type == 'c':
                m_jaw_width = m_jaw_width - .03959

            if g == 'l':
                gripper_m.lg_jaw_to(jaw_width)
                m_rotmat = hnd_rotmat
                m_pos = np.array([-.053 - m_jaw_width / 2, .018, -.135])
                m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
            elif g == 'r':
                gripper_m.rg_jaw_to(jaw_width)
                m_rotmat = hnd_rotmat
                m_pos = np.array([.053 + m_jaw_width / 2, -.018, -.135])
                m_pos = hnd_pos + hnd_rotmat.dot(m_pos)

            gripper_m.fix_to(m_pos, m_rotmat)
            gripper_m.mg_jaw_to(m_jaw_width)
            # gripper_m.gen_meshmodel().attach_to(base)

            if g == 'l':
                gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="l")
                finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
                finger_2.set_pos(finger_2_pos)
                finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
                finger_2.set_rotmat(finger_2_rotmat)
                # finger_1.attach_to(base)
                # finger_2.attach_to(base)
            elif g == 'r':
                gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="r")
                finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
                finger_2.set_pos(finger_2_pos)
                finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
                finger_2.set_rotmat(finger_2_rotmat)
                # finger_1.attach_to(base)
                # finger_2.attach_to(base)
            gripper_m.gripper_bool(g='l')
            gripper_m.gripper_bool(g='r')
            gripper_m.gripper_bool(g='m')

            gripper_b.grip_at_with_jcpose(m_jaw_center_pos, m_jaw_center_rotmat, m_jaw_width)
            a = gripper_b.pos
            b = gripper_b.rotmat
            gripper_m.fix_to(a, b)
            gripper_m.mg_jaw_to(m_jaw_width)

            c = np.dot(b, m_rotmat.T)
            finger_1_pos_new = np.dot(c, np.array([0, 0, 0]) - m_pos) + a
            finger_1_rotmat_new = c
            finger_1_new.set_pos(finger_1_pos_new)
            finger_1_new.set_rotmat(finger_1_rotmat_new)
            objcm_1 = cm.CollisionModel(finger_2_new, cdprimit_type='box', expand_radius=.1)

            finger_2_pos_new = np.dot(c, finger_2_pos - m_pos) + a
            finger_2_rotmat_new = np.dot(c, finger_2_rotmat)
            finger_2_new.set_pos(finger_2_pos_new)
            finger_2_new.set_rotmat(finger_2_rotmat_new)
            objcm_2 = cm.CollisionModel(finger_2_new, cdprimit_type='box', expand_radius=.1)

            if not target_object.is_mcdwith([finger_2_new, finger_1_new]):
                finger_1_new.attach_to(base)
                finger_2_new.attach_to(base)
                gripper_m.gen_meshmodel().attach_to(base)


base.run()
