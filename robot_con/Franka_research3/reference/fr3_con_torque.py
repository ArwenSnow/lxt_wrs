#!/usr/bin/env python3


# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Joint Impedance Target Example

This example shows a joint impedance controller that renders a spring damper system
to move the robot through a sequence of target joint configurations.
The controller will generate smooth trajectories between positions and run in a continuous loop.
"""

import argparse
import sys
import time

import numpy as np

from pylibfranka import Robot, Torques
import robot_sim.robots.Franka_research3.Franka_research3 as fr3
import motion.probabilistic.rrt_connect as rrtc
import motion.trajectory.piecewisepoly_toppra as pwp


class SimpleMotionGenerator:
    """Simple minimum jerk trajectory generator for smooth joint motion."""

    def __init__(self, start_position, end_position, duration=3.0):
        """Initialize the trajectory generator.

        Args:
            start_position: Starting joint positions (array of 7 values)
            end_position: Target joint positions (array of 7 values)
            duration: Duration of the trajectory in seconds
        """
        self.start_position = np.array(start_position)
        self.end_position = np.array(end_position)
        self.duration = duration
        self.start_time = None

    def start(self):
        """Start the trajectory."""
        self.start_time = time.time()

    def get_position(self):
        """Get the current target position along the trajectory."""
        if self.start_time is None:
            return self.start_position

        elapsed_time = time.time() - self.start_time
        s = self._minimum_jerk(min(elapsed_time / self.duration, 1.0))

        return self.start_position + s * (self.end_position - self.start_position)

    def is_finished(self):
        """Check if the trajectory is complete."""
        if self.start_time is None:
            return False

        elapsed_time = time.time() - self.start_time
        return elapsed_time >= self.duration

    def _minimum_jerk(self, t):
        """Minimum jerk trajectory profile (normalized [0,1])."""
        return 10 * (t**3) - 15 * (t**4) + 6 * (t**5)


def main(target_joint_positions):
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()

    # # Define a sequence of target joint configurations
    # target_joint_positions = [
    #     # Home position (slightly bent arm)
    #     [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
    #     # Extended arm pointing forward
    #     [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0],
    #     # Arm pointing to the right
    #     [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
    #     # Arm pointing to the left
    #     [-0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
    #     # Home position again
    #     [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
    # ]

    # Compliance parameters
    joint_stiffness = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]

    try:
        # Connect to robot
        robot = Robot(args.ip)

        # Set default behavior
        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        # Get initial state
        initial_state = robot.read_once()
        current_position = np.array(initial_state.q)

        # Start torque control
        active_control = robot.start_torque_control()

        # Create a model instance from robot
        model = robot.load_model()

        # Main control variables
        wait_time = 0.5  # Time to wait at each position before moving to next

        # For loop over target positions
        for target_position in target_joint_positions:
            # Initialize trajectory for this target
            trajectory = SimpleMotionGenerator(
                current_position,
                target_position,
                duration=3.0,
            )
            trajectory.start()

            # Control until we reach target and wait time is complete
            target_reached = False
            wait_started = False
            wait_start_time = 0
            firsttime = True

            # Control loop for current trajectory
            while True:
                # Read robot state
                robot_state, _ = active_control.readOnce()

                # Get state variables
                coriolis = np.array(model.coriolis(robot_state))
                q = np.array(robot_state.q)
                dq = np.array(robot_state.dq)
                if firsttime:
                    dq_filtered = dq
                    firsttime = False
                else:
                    dq_filtered = (1 - 0.2) * dq_filtered + 0.2 * dq

                # Get current target from trajectory
                q_goal = trajectory.get_position()

                # Compute error to desired equilibrium joint configuration
                position_error = q - q_goal

                # Compute joint-space impedance control
                tau_task = np.zeros(7)
                for i in range(7):
                    tau_task[i] = -joint_stiffness[i] * position_error[i] - joint_damping[i] * dq_filtered[i]  # 阻抗控制律

                # Add coriolis compensation
                tau_d = tau_task + coriolis                                                           # 添加补偿

                # Convert to array for Torques command
                torque_command = Torques(tau_d.tolist())
                torque_command.motion_finished = False
                active_control.writeOnce(torque_command)

                # Check if trajectory is finished
                if trajectory.is_finished() and not target_reached:
                    target_reached = True
                    wait_started = True
                    wait_start_time = time.time()

                # Check if we've waited long enough
                if wait_started and (time.time() - wait_start_time >= wait_time):
                    # Update current position for next trajectory
                    current_position = q_goal
                    break

    except Exception as e:
        print(f"\nError occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1

    return 0

if __name__ == "__main__":
    robot_s = fr3.Franka_research3()
    rrtc_planner = rrtc.RRTConnect(robot_s)
    obstacle_list = []

    start_conf = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0])
    goal_conf = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, np.pi/6])

    grasp_path = rrtc_planner.plan(component_name="arm",
                                   start_conf=start_conf,
                                   goal_conf=goal_conf,
                                   obstacle_list=obstacle_list,
                                   ext_dist=0.001,
                                   max_time=300)

    tpply = pwp.PiecewisePolyTOPPRA()
    interpolated_path = tpply.interpolate_by_max_spdacc(path=grasp_path,
                                                        control_frequency=.002,
                                                        max_vels=[1, 1, 1, 1, 1, 1, 1],
                                                        max_accs=[1] * 7,
                                                        toggle_debug=False)
    main(interpolated_path.tolist())
