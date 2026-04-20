import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.robots.Franka_research3.Franka_research3 as Fr
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rg


# world
base = wd.World(cam_pos=[-2, 4, 1.5], lookat_pos=[0, 0, 0])
# gm.gen_frame().attach_to(base)

# gripper
lft_gripper = rg.Reconfgripper().lft
rgt_gripper = rg.Reconfgripper().rgt
gripper = rg.Reconfgripper()

# finger
finger_1 = cm.CollisionModel("../objects/finger_b_2.stl")
finger_1_pos = np.array([0, .065, -.09398])
finger_1_rotmat = rm.rotmat_from_axangle([0, 0, 1], -math.pi/2)
finger_1.set_pos(finger_1_pos)
finger_1.set_rotmat(finger_1_rotmat)
# gm.gen_frame(finger_1_pos, finger_1_rotmat).attach_to(base)
finger_1.set_rgba([.9, .75, .35, 1])
finger_1.attach_to(base)

# tool_b_2 for planning first finger grasps
tool_1 = cm.CollisionModel("../objects/tool_b_2.stl")
tool_pos = np.array([0, 0, 0])
tool_rotmat = np.eye(3)
tool_1.set_pos(tool_pos)
tool_1.set_rotmat(tool_rotmat)
gm.gen_frame(np.array([0, 0, 0]), np.eye(3)).attach_to(base)

# step1 : planning for first
first_grasp_list = gpa.plan_grasps(lft_gripper, tool_1,
                                   angle_between_contact_normals=math.radians(160),
                                   openning_direction='loc_x',
                                   max_samples=10, min_dist_between_sampled_contact_points=.005,
                                   contact_offset=.001)

objcm_list = [finger_1]
lft_grasp_info_list = []
rgt_grasp_info_list = []
for grasp_info in first_grasp_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    x_axis_direction = jaw_center_rotmat[:, 0]
    desired_direction = tool_rotmat[:, 1]
    cos_angle = (np.dot(x_axis_direction, desired_direction) /
                 (np.linalg.norm(x_axis_direction) * np.linalg.norm(desired_direction)))
    if cos_angle == 1 or cos_angle == -1:
        lft_gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        l_pos = lft_gripper.pos
        l_rotmat = lft_gripper.rotmat
        g_pos_lft = l_rotmat@np.array([-.053, .018, -.132]) + l_pos
        g_rotmat_lft = l_rotmat
        gripper.fix_to(g_pos_lft, g_rotmat_lft)
        gripper.lg_jaw_to(jaw_width)
        if not gripper.is_mesh_collided(objcm_list=objcm_list):
            # gripper.gen_meshmodel().attach_to(base)
            lft_grasp_info_list.append(grasp_info)

        rgt_gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        r_pos = rgt_gripper.pos
        r_rotmat = rgt_gripper.rotmat
        g_pos_rgt = r_rotmat@np.array([.053, -.018, -.132]) + r_pos
        g_rotmat_rgt = r_rotmat
        gripper.fix_to(g_pos_rgt, g_rotmat_rgt)
        gripper.rg_jaw_to(jaw_width)
        if not gripper.is_mesh_collided(objcm_list=objcm_list):
            # gripper.gen_meshmodel().attach_to(base)
            rgt_grasp_info_list.append(grasp_info)

gpa.write_pickle_file('finger_1', lft_grasp_info_list, './', 'first_lft_grasps.pickle')
gpa.write_pickle_file('finger_2', rgt_grasp_info_list, './', 'first_rgt_grasps.pickle')

base.run()





