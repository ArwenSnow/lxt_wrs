import os
import math
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi

class PGC(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(3), name='PGC', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[2]['loc_pos'] = np.array([.0626,-.4,.1634])
        self.jlc.jnts[2]['loc_motionax'] = np.array([1, 0, 0])

        # links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "PGC.STL")
        self.jlc.lnks[0]['rgba'] = [.2,.2,.2, 1]
        self.jlc.reinitialize()

        # collision checker
        if enable_cc:
            super().enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[0],
                    self.jlc.lnks[1]]
        intolist = [self.jlc.lnks[3],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[2]]
        intolist = [self.jlc.lnks[4],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[3]]
        intolist = [self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = PGC(enable_cc=True)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    # manipulator_meshmodel.show_cdprimit()
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    base.run()