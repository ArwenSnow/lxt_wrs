#!/usr/bin/env python3
# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

import argparse
import time
from pylibfranka import Gripper


parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
parser.add_argument("--width", type=float, default=0.005, help="Object width to grasp")
parser.add_argument("--homing", type=int, default=1, choices=[0, 1], help="Perform homing (0 or 1)")
parser.add_argument("--speed", type=float, default=0.1, help="Gripper speed")
parser.add_argument("--force", type=float, default=60, help="Gripper force")
args = parser.parse_args()

try:
    # Connect to gripper
    gripper = Gripper(args.ip)
    grasping_width = args.width

    if args.homing:
        gripper.homing()      # 初始化以测量当前抓取行程
    if not gripper.grasp(grasping_width, args.speed, args.force):  # 抓物体
        print("Failed to grasp object.")
except Exception as e:
    print(f"Error oc curre  d: {e}")