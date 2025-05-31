import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh60.dh60 as dh


def classify(grasp_info_list, object_name, obj_finger_list):
    finger_list = {f"finger{i}": [] for i in range(1, 9)}
    finger_counts = {f"finger{i}": 0 for i in range(1, 9)}
    for grasp_info in grasp_info_list:
        jaw_width = grasp_info[0]
        if 0 <= jaw_width <= 0.063:
            finger_list["finger4"].append(grasp_info)
            finger_counts["finger4"] += 1
        elif 0.074 <= jaw_width <= 0.137:
            finger_list["finger5"].append(grasp_info)
            finger_counts["finger5"] += 1
        elif 0.138 <= jaw_width <= 0.201:
            finger_list["finger6"].append(grasp_info)
            finger_counts["finger6"] += 1
        elif 0.183 <= jaw_width <= 0.246:
            finger_list["finger7"].append(grasp_info)
            finger_counts["finger7"] += 1
        elif 0.293 <= jaw_width <= 0.356:
            finger_list["finger8"].append(grasp_info)
            finger_counts["finger8"] += 1
    used_fingers = []
    for finger_name, count in finger_counts.items():
        if count > 0:
            finger_num = int(finger_name.replace("finger", ""))
            used_fingers.append(finger_num)
    obj_finger_list[object_name] = used_fingers
    return finger_list, finger_counts


base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# gripper
gripper_s = dh.Dh60()

# objects
screw = cm.CollisionModel("objects/screw.stl")
screw.set_rgba([.9, .75, .35, 1])
screw.set_pos(np.array([0, 0, 0]))

milk_box = cm.CollisionModel("objects/milk_box.stl")
milk_box.set_rgba([.9, .75, .35, 1])
milk_box.set_pos(np.array([0, 0, 0]))

box = cm.CollisionModel("objects/box.stl")
box.set_rgba([.9, .75, .35, 1])
box.set_pos(np.array([0, 0, 0]))

object_finger_list = {
    "screw": [],
    "milk_box": [],
    "gift": [],
    "box": [],
    "square_bottle": [],
    "strawberry": [],
    "pyramid": []
}

grasp_screw_info_list = gpa.plan_grasps(gripper_s, screw,
                                        angle_between_contact_normals=math.radians(160),
                                        openning_direction='loc_x',
                                        max_samples=100, min_dist_between_sampled_contact_points=.005,
                                        contact_offset=.001)

grasp_milk_box_info_list = gpa.plan_grasps(gripper_s, milk_box,
                                           angle_between_contact_normals=math.radians(160),
                                           openning_direction='loc_x',
                                           max_samples=100, min_dist_between_sampled_contact_points=.005,
                                           contact_offset=.001)

grasp_box_info_list = gpa.plan_grasps(gripper_s, box,
                                      angle_between_contact_normals=math.radians(160),
                                      openning_direction='loc_x',
                                      max_samples=100, min_dist_between_sampled_contact_points=.005,
                                      contact_offset=.001)


classify(grasp_screw_info_list, "screw", object_finger_list)
classify(grasp_milk_box_info_list, "milk_box", object_finger_list)
classify(grasp_box_info_list, "box", object_finger_list)
print(object_finger_list)


base.run()
