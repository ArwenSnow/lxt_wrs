import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xc

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# object
object = cm.CollisionModel("objects/finger.stl")
object.set_rgba([.9, .75, .35, 1])
object.attach_to(base)

# hnd_s
gripper_s = xc.xc330gripper()

# grasp_info
jaw_center_pos = np.zeros(3)
jaw_center_rotmat = np.eye(3)
contact_offset = .0025
jaw_width = .008 + contact_offset*2
angle_increment = math.pi*45/180
grasp_info_list = []

for i in range(8):
    jaw_center_rotmat = np.dot(rm.rotmat_from_axangle([1, 0, 0], i*angle_increment ),jaw_center_rotmat)
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    grasp_info_list.append({
        'position': jaw_center_pos,
        'rotation': jaw_center_rotmat,
        'width': jaw_width
    })
base.run()

print("grasp_info_list:")
for info in grasp_info_list:
    print(f"Position: {info['position']}, Rotation: {info['rotation']}, Width: {info['width']}")

