#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

# !/usr/bin/env python3

import argparse
import time
import numpy as np
from pylibfranka import ControllerMode, JointPositions, Robot

def fr3_position_control(robot_s,
                         rrtc_planner,
                         obstacle_list,
                         tpply,
                         tgt_pos,
                         tgt_rotmat):
    # Connect to robot
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()
    robot = Robot(args.ip)

    try:
        # Set collision behavior
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        # Start joint position control with external control loop
        active_control = robot.start_joint_position_control(ControllerMode.CartesianImpedance)

        initial_position = [0.0] * 7
        time_elapsed = 0.0
        motion_finished = False
        idx = 0

        robot_state, duration = active_control.readOnce()
        initial_position = robot_state.q_d if hasattr(robot_state, "q_d") else robot_state.q
        robot_s.fk("arm", np.array(initial_position))

        goal_conf = robot_s.ik("arm", tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)

        grasp_path = rrtc_planner.plan(component_name="arm",
                                       start_conf=np.array(initial_position),
                                       goal_conf=goal_conf,
                                       obstacle_list=obstacle_list,
                                       ext_dist=0.001,
                                       max_time=300)
        interpolated_path = tpply.interpolate_by_max_spdacc(path=grasp_path,
                                                            control_frequency=.001,
                                                            max_vels=[1, 1, 1, 1, 1, 1, 1],
                                                            max_accs=[1] * 7,
                                                            toggle_debug=False)

        # First move the robot to a suitable joint configuration
        print("WARNING: This example will move the robot!")
        print("Please make sure to have the user stop button at hand!")
        input("Press Enter to continue...")

        # External control loop
        while not motion_finished:
            # # Read robot state and duration
            # robot_state, duration = active_control.readOnce()
            new_positions = interpolated_path[idx]

            idx += 1

            # Set joint positions
            joint_positions = JointPositions(new_positions)
            print("当前时间为：", time_elapsed, "，当前关节角度为：", new_positions)

            if idx >= len(interpolated_path):
                joint_positions.motion_finished = True
                motion_finished = True
                print("Finished motion, shutting down example")

            # Send command to robot
            active_control.writeOnce(joint_positions)

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1



if __name__ == "__main__":
    import basis.robot_math as rm
    import robot_sim.robots.Franka_research3.Franka_research3 as fr3
    import motion.probabilistic.rrt_connect as rrtc
    import motion.trajectory.piecewisepoly_toppra as pwp
    import math

    robot_s = fr3.Franka_research3()
    rrtc_planner = rrtc.RRTConnect(robot_s)
    obstacle_list = []
    tpply = pwp.PiecewisePolyTOPPRA()
    tgt_pos = np.array([.4, .1, .4])
    tgt_rot = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)

    fr3_position_control(robot_s = robot_s,rrtc_planner=rrtc_planner,obstacle_list=obstacle_list,tpply=tpply, tgt_pos=tgt_pos, tgt_rotmat=tgt_rot)



