import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi
from panda3d.core import CollisionNode, CollisionBox, Point3


class Pca(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='pca', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)

        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), n_links = 6+1
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 79.6/1000])
        self.jlc.jnts[1]['loc_rotmat'] = np.eye(3)
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[2]['loc_pos'] = np.array([0, 117.99/1000, 138.5/1000])
        z_2 = rm.rotmat_from_axangle([1, 0, 0], -math.pi/2)
        y_2 = rm.rotmat_from_axangle([0, 1, 0], -math.pi/2)
        self.jlc.jnts[2]['loc_rotmat'] = y_2@z_2
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[3]['loc_pos'] = np.array([0, -101/1000, 0])
        self.jlc.jnts[3]['loc_rotmat'] = rm.rotmat_from_axangle([1, 0, 0], -math.pi/2)
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 0, -1])

        self.jlc.jnts[4]['loc_pos'] = np.array([0, 0, -154/1000])
        self.jlc.jnts[4]['loc_rotmat'] = rm.rotmat_from_axangle([1, 0, 0], math.pi/2)
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[5]['loc_pos'] = np.array([0, -47.65/1000, 0])
        z_5 = rm.rotmat_from_axangle([0, 0, 1], -math.pi/2)
        y_5 = rm.rotmat_from_axangle([0, 1, 0], -math.pi/2)
        self.jlc.jnts[5]['loc_rotmat'] = z_5@y_5
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 0, -1])

        self.jlc.jnts[6]['loc_pos'] = np.array([-93.75/1000, -54.4/1000, -251.85/1000])
        z_6 = rm.rotmat_from_axangle([0, 0, 1], -math.pi/2)
        y_6 = rm.rotmat_from_axangle([0, 1, 0], -math.pi/2)
        self.jlc.jnts[6]['loc_rotmat'] = z_6@y_6
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[7]['loc_pos'] = np.array([0, 65/1000, 78/1000])
        z_7 = rm.rotmat_from_axangle([0, 0, 1], math.pi/2)
        y_7 = rm.rotmat_from_axangle([0, 1, 0], -math.pi/2)
        self.jlc.jnts[7]['loc_rotmat'] = z_7@y_7
        self.jlc.jnts[7]['loc_motionax'] = np.array([0, 0, 1])

        # links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK00.STL")
        self.jlc.lnks[0]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "LINK00.STL"),
                                                                cdprimit_type="user_defined",
                                                                userdefined_cdprimitive_fn=self._link_00)
        self.jlc.lnks[0]['rgba'] = [.2, .2, .2, 1]

        self.jlc.lnks[1]['name'] = "shoulder"
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK00.STL")
        self.jlc.lnks[1]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "LINK01.STL"),
                                                                cdprimit_type="user_defined",
                                                                userdefined_cdprimitive_fn=self._link_01)
        self.jlc.lnks[1]['rgba'] = [.2, .2, .2, 1]

        self.jlc.lnks[2]['name'] = "upperarm"
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK01.STL")
        self.jlc.lnks[2]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "LINK02.STL"),
                                                                cdprimit_type="user_defined",
                                                                userdefined_cdprimitive_fn=self._link_02)
        self.jlc.lnks[2]['rgba'] = [.2, .2, .2, 1]

        self.jlc.lnks[3]['name'] = "forearm"
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK02.STL")
        self.jlc.lnks[3]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "LINK03.STL"),
                                                                cdprimit_type="user_defined",
                                                                userdefined_cdprimitive_fn=self._link_03)
        self.jlc.lnks[3]['rgba'] = [.2, .2, .2, 1]

        self.jlc.lnks[4]['name'] = "wrist1"
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK03.STL")
        self.jlc.lnks[4]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "LINK04.STL"),
                                                                cdprimit_type="user_defined",
                                                                userdefined_cdprimitive_fn=self._link_04)
        self.jlc.lnks[4]['rgba'] = [.2, .2, .2, 1]

        self.jlc.lnks[5]['name'] = "wrist2"
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK04.STL")
        self.jlc.lnks[5]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "LINK05.STL"),
                                                                cdprimit_type="user_defined",
                                                                userdefined_cdprimitive_fn=self._link_05)
        self.jlc.lnks[5]['rgba'] = [.7, .7, .7, 1]

        self.jlc.lnks[6]['name'] = "wrist3"
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK05.STL")
        self.jlc.lnks[6]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "LINK06.STL"),
                                                                cdprimit_type="user_defined",
                                                                userdefined_cdprimitive_fn=self._link_06)
        self.jlc.lnks[6]['rgba'] = [.7, .7, .7, 1]

        self.jlc.lnks[7]['name'] = "wrist4"
        self.jlc.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "LINK06.STL")
        self.jlc.lnks[7]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "LINK07.STL"),
                                                                cdprimit_type="user_defined",
                                                                userdefined_cdprimitive_fn=self._link_07)
        self.jlc.lnks[7]['rgba'] = [.7, .7, .7, 1]

        self.jlc.reinitialize()

        if enable_cc:
            self.enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6, 7])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6],
                      self.jlc.lnks[7]]
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

    @staticmethod
    def _link_00(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(.0, .0, .038),
                                              x=.026 + radius, y=.026 + radius, z=.038 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    @staticmethod
    def _link_01(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(.0, .0368, -.0035),
                                              x=.02 + radius, y=.055 + radius, z=.01 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(.0, .118, .08),
                                              x=.03 + radius, y=.04 + radius, z=.09 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    @staticmethod
    def _link_02(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.0435, .0, .0),
                                              x=.06 + radius, y=.032 + radius, z=.035 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    @staticmethod
    def _link_03(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(.0, .0, -.09),
                                              x=.031 + radius, y=.026 + radius, z=.09 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    @staticmethod
    def _link_04(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.05, .0, .0),
                                              x=.07 + radius, y=.025 + radius, z=.035 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    @staticmethod
    def _link_05(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.08, .0, -.064),
                                              x=.09 + radius, y=0.06 + radius, z=.035 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.09375, -.08, -.175),
                                              x=.04 + radius, y=.04 + radius, z=.1 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    @staticmethod
    def _link_06(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(.032, .0, .0),
                                              x=.03 + radius, y=.012 + radius, z=.009 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(.064, .0, .05),
                                              x=.015 + radius, y=.012 + radius, z=.045 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    @staticmethod
    def _link_07(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(.0, .0, -.0282),
                                              x=.02 + radius, y=.02 + radius, z=.029 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    Pca_manipulator = Pca(enable_cc=True)
    manipulator_meshmodel = Pca_manipulator.gen_meshmodel(toggle_jntscs=True, toggle_tcpcs=False)
    manipulator_meshmodel.attach_to(base)

    # Pca_manipulator.show_cdprimit()

    base.run()
