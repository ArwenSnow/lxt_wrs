import os
import math
import numpy as np
import modeling.model_collection as mc
import modeling.collision_model as cm
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp


class PcaHandle(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3), cdmesh_type='box', name='Pca_handle', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.coupling.jnts[1]['loc_pos'] = coupling_offset_pos
        self.coupling.jnts[1]['loc_rotmat'] = coupling_offset_rotmat
        self.coupling.lnks[0]['collision_model'] = cm.gen_stick(self.coupling.jnts[0]['loc_pos'],
                                                                self.coupling.jnts[1]['loc_pos'],
                                                                thickness=.07, rgba=[.2, .2, .2, 1],
                                                                sections=24)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']

        # lft
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='handle_0_1')

        self.lft.lnks[0]['name'] = "handle_0"
        self.lft.lnks[0]['mesh_file'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "handle_0.stl"), cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self.handle_0, expand_radius=.000)
        self.lft.lnks[0]['rgba'] = [.5, .5, .5, 1]

        self.lft.jnts[1]['loc_pos'] = np.array([-67.5/1000, .0, 66.5/1000])
        self.lft.jnts[1]['loc_rotmat'] = rm.rotmat_from_axangle([0, 1, 0], math.pi/2)

        self.lft.lnks[1]['name'] = "handle_1"
        self.lft.lnks[1]['mesh_file'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "handle_1.stl"), cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self.handle_1, expand_radius=.000)
        self.lft.lnks[1]['rgba'] = [.5, .5, .5, 1]

        # rgt
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='handle_2')

        self.rgt.jnts[1]['loc_pos'] = np.array([36.39/1000, 37.77/1000, 69.01/1000])
        self.rgt.jnts[1]['loc_rotmat'] = rm.rotmat_from_axangle([0, 1, 0], math.pi/2)
        self.rgt.jnts[1]['type'] = 'revolute'
        self.rgt.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt.jnts[1]['motion_rng'] = [-math.radians(10), math.radians(10)]

        self.rgt.lnks[1]['name'] = "handle_2"
        self.rgt.lnks[1]['mesh_file'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "handle_2.stl"), cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self.handle_2, expand_radius=.000)
        self.rgt.lnks[1]['rgba'] = [.5, .5, .5, 1]

        # reinitialize
        self.lft.reinitialize()
        self.rgt.reinitialize()

        # jaw width
        self.jawwidth_rng = [0.0, 600]

        # # jaw center
        # self.jaw_center_pos = np.array([76.75/1000, 42.41/1000, 66.51/1000]) + coupling_offset_pos
        # self.jaw_center_rotmat = (rm.rotmat_from_axangle([0, 1, 0], math.pi/2) @
        #                           rm.rotmat_from_axangle([1, 0, 0], -math.pi/2))

        # jaw center for vision
        self.jaw_center_pos = np.array([98.55/1000, 0, 66.51/1000]) + coupling_offset_pos
        self.jaw_center_rotmat = (rm.rotmat_from_axangle([0, 1, 0], math.pi/2) @
                                  rm.rotmat_from_axangle([1, 0, 0], -math.pi/2))

        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    @staticmethod
    def handle_0(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.025, .0, .007),
                                              x=.04 + radius, y=0.013 + radius, z=.01 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.065, .0, .042),
                                              x=.014 + radius, y=0.014 + radius, z=.045 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    @staticmethod
    def handle_1(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(.0, -.005, .07),
                                              x=.02 + radius, y=0.024 + radius, z=.08 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(.0, .007, .15),
                                              x=.025 + radius, y=0.037 + radius, z=.035 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    @staticmethod
    def handle_2(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(.0025, -.025, .01),
                                              x=.003 + radius, y=0.03 + radius, z=.02 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft, [0, 1])
            self.cc.add_cdlnks(self.rgt, [1])
            activelist = [self.lft.lnks[0],
                          self.lft.lnks[1],
                          self.rgt.lnks[1]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: motion_val, meter or radian
        """
        if self.rgt.jnts[1]['motion_rng'][0] <= motion_val <= self.rgt.jnts[1]['motion_rng'][1]:
            self.rgt.jnts[1]['motion_val'] = motion_val
            self.rgt.fk()
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def turn_on(self, radian):
        """
        Press the switch on the handle
        """
        if self.rgt.jnts[1]['motion_rng'][0] <= radian <= self.rgt.jnts[1]['motion_rng'][1]:
            self.fk(motion_val=radian)
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def jaw_to(self, jaw_width):
        if jaw_width > self.jawwidth_rng[1]:
            raise ValueError("The jaw_width parameter is out of range!")
        self.fk(motion_val=jaw_width / 2.0)

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='pca_handle_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(stickmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)

        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='pca_handle_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.lft.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.rgt.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(meshmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    gm.gen_frame().attach_to(base)

    grpr = PcaHandle(enable_cc=True)
    grpr.show_cdprimit()

    # grpr.turn_on(math.radians(5))
    grpr.gen_meshmodel(toggle_tcpcs=True).attach_to(base)

    base.run()

