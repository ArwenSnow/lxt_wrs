if __name__ == '__main__':
    import math
    import numpy as np
    import basis.robot_math as rm
    import robot_sim.robots.lxt_robot.lxt_robot as rbt
    import motion.probabilistic.rrt_connect as rrtc
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import time

    start = time.time()
    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    robot_s = rbt.bigGripper()
    robot_s.gen_meshmodel().attach_to(base)
    start_conf = robot_s.get_jnt_values(component_name='arm')
    jnt_values = start_conf + np.array([0, 0, .08,0,0,0])
    robot_s.fk(component_name="arm", jnt_values=jnt_values)
    robot_s.gen_meshmodel(rgba=(0,0,1,1)).attach_to(base)
    base.run()