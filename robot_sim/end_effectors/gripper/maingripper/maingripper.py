import os
import math
import numpy as np
import modeling.model_collection as mc
import modeling.collision_model as cm
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp


class maingripper(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='convex_hull', name='maingripper',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']

        # gripper base
        self.body = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(7), name='base')
        self.body.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.body.lnks[0]['name'] = "PGC_300_60_W_S"
        self.body.lnks[0]['loc_pos'] = np.zeros(3)
        self.body.lnks[0]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "PGC_300_60_W_S.stl"),
                                                                 expand_radius=.001)
        self.body.lnks[0]['rgba'] = [0.2, 0.2, 0.2, 1]

        # lft
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(11), name='lft')
        self.lft.jnts[1]['loc_pos'] = np.array([.0062, -.01635, .143])
        self.lft.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0,0)
        self.lft.jnts[1]['type'] = 'prismatic'
        self.lft.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[2]['loc_pos'] = np.array([.0264, .01635, -.0027])
        self.lft.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[3]['loc_pos'] = np.array([0, 0, .0231])
        self.lft.jnts[3]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[4]['loc_pos'] = np.array([0, 0, 0])
        self.lft.jnts[4]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[5]['loc_pos'] = np.array([0, 0, .003])
        self.lft.jnts[5]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[6]['loc_pos'] = np.array([-.0015, -0.0066, 0.00505])
        self.lft.jnts[6]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[7]['loc_pos'] = np.array([0, 0.0132, 0])
        self.lft.jnts[7]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[8]['loc_pos'] = np.array([-.000286, .004, .00594])
        self.lft.jnts[8]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[9]['loc_pos'] = np.array([0, 0, 0])
        self.lft.jnts[9]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[10]['loc_pos'] = np.array([.004, -.0212, 0])
        self.lft.jnts[10]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[11]['loc_pos'] = np.array([0, 0, 0])
        self.lft.jnts[11]['loc_motionax'] = np.array([-1, 0, 0])


        self.lft.lnks[1]['name'] = "zhijianzujian1"
        self.lft.lnks[1]['loc_pos'] = np.array([0,0,0])
        self.lft.lnks[1]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "zhijian.stl"),
                                                                expand_radius=.001)
        self.lft.lnks[1]['rgba'] = [.57, .57, .57, 1]
        self.lft.lnks[2]['name'] = "gripper_base1"
        self.lft.lnks[2]['loc_pos'] = np.array([0, 0, 0])
        self.lft.lnks[2]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "gripper_base.stl"),
                                                                expand_radius=.001)
        self.lft.lnks[2]['rgba'] = [.57, .57, .57, 1]
        self.lft.lnks[3]['name'] = "dianji1"
        self.lft.lnks[3]['loc_pos'] = np.array([0, 0, 0])
        self.lft.lnks[3]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "dianji.stl"),
                                                                expand_radius=.001)
        self.lft.lnks[3]['rgba'] = [.2, .2, .2, 1]
        self.lft.lnks[4]['name'] = "base1"
        self.lft.lnks[4]['loc_pos'] = np.array([0, 0, 0])
        self.lft.lnks[4]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "base.stl"),
                                                                expand_radius=.001)
        self.lft.lnks[4]['rgba'] = [.37, .37, .37, 1]
        self.lft.lnks[5]['name'] = "chilun1"
        self.lft.lnks[5]['loc_pos'] = np.array([0, 0, 0])
        self.lft.lnks[5]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "chilun.stl"),
                                                                expand_radius=.001)
        self.lft.lnks[5]['rgba'] = [.37, .37, .37, 1]
        self.lft.lnks[6]['name'] = "huagui1"
        self.lft.lnks[6]['loc_pos'] = np.array([0, 0, 0])
        self.lft.lnks[6]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huagui.stl"),
                                                                 expand_radius=.001)
        self.lft.lnks[6]['rgba'] = [.57, .57, .57, 1]
        self.lft.lnks[7]['name'] = "huagui2"
        self.lft.lnks[7]['loc_pos'] = np.array([0, 0, 0])
        self.lft.lnks[7]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huagui.stl"),
                                                                expand_radius=.001)
        self.lft.lnks[7]['rgba'] = [.57, .57, .57, 1]
        self.lft.lnks[8]['name'] = "finger1"
        self.lft.lnks[8]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "gripper1.stl"),
                                                          expand_radius=.001)
        self.lft.lnks[8]['rgba'] = [.2, .2, .2, 1]
        self.lft.lnks[9]['name'] = "huakuai1"
        self.lft.lnks[9]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huakuai1.stl"),
                                                          expand_radius=.001)
        self.lft.lnks[9]['rgba'] = [.6, .6, .6, 1]
        self.lft.lnks[10]['name'] = "finger2"
        self.lft.lnks[10]['loc_pos'] = np.array([0, 0, 0])
        self.lft.lnks[10]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "gripper2.stl"),
                                                           expand_radius=.001)
        self.lft.lnks[10]['rgba'] = [.2, .2, .2, 1]
        self.lft.lnks[11]['name'] = "huakuai2"
        self.lft.lnks[11]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huakuai2.stl"),
                                                           expand_radius=.001)
        self.lft.lnks[11]['rgba'] = [.6, .6, .6, 1]


        # rgt
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(11), name='rgt')
        self.rgt.jnts[1]['loc_pos'] = np.array([-.0062,.01635,.143])
        self.rgt.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0,0)
        self.rgt.jnts[1]['type'] = 'prismatic'
        self.rgt.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[2]['loc_pos'] = np.array([-.0264, -.01635, -.0027])
        self.rgt.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[3]['loc_pos'] = np.array([0, 0, .0231])
        self.rgt.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[4]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.jnts[4]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[5]['loc_pos'] = np.array([0, 0, .003])
        self.rgt.jnts[5]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[6]['loc_pos'] = np.array([0.0015, -0.0066, 0.00505])
        self.rgt.jnts[6]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[7]['loc_pos'] = np.array([0, 0.0132, 0])
        self.rgt.jnts[7]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[8]['loc_pos'] = np.array([-.000286, .004, .00594])
        self.rgt.jnts[8]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[9]['loc_pos'] = np.array([0,0,0])
        self.rgt.jnts[9]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[10]['loc_pos'] = np.array([.004, -.0212, 0])
        self.rgt.jnts[10]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.jnts[11]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.jnts[11]['loc_motionax'] = np.array([1, 0, 0])

        self.rgt.lnks[1]['name'] = "zhijianzujian2"
        self.rgt.lnks[1]['loc_pos'] = np.array([0,0,0])
        self.rgt.lnks[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi *1)
        self.rgt.lnks[1]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "zhijian.stl"),
                                                                expand_radius=.001)
        self.rgt.lnks[1]['rgba'] = [.57, .57, .57, 1]
        self.rgt.lnks[2]['name'] = "gripper_base2"
        self.rgt.lnks[2]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.lnks[2]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi * 1)
        self.rgt.lnks[2]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "gripper_base.stl"),
                                                                 expand_radius=.001)
        self.rgt.lnks[2]['rgba'] = [.57, .57, .57, 1]
        self.rgt.lnks[3]['name'] = "dianji2"
        self.rgt.lnks[3]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.lnks[3]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi * 1)
        self.rgt.lnks[3]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "dianji.stl"),
                                                                 expand_radius=.001)
        self.rgt.lnks[3]['rgba'] = [.2, .2, .2, 1]
        self.rgt.lnks[4]['name'] = "base2"
        self.rgt.lnks[4]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.lnks[4]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi * 1)
        self.rgt.lnks[4]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "base.stl"),
                                                                 expand_radius=.001)
        self.rgt.lnks[4]['rgba'] = [.37, .37, .37, 1]
        self.rgt.lnks[5]['name'] = "chilun2"
        self.rgt.lnks[5]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.lnks[5]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi * 1)
        self.rgt.lnks[5]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "chilun.stl"),
                                                                 expand_radius=.001)
        self.rgt.lnks[5]['rgba'] = [.37, .37, .37, 1]
        self.rgt.lnks[6]['name'] = "huagui3"
        self.rgt.lnks[6]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.lnks[6]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huagui.stl"),
                                                                 expand_radius=.001)
        self.rgt.lnks[6]['rgba'] = [.57, .57, .57, 1]
        self.rgt.lnks[7]['name'] = "huagui4"
        self.rgt.lnks[7]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.lnks[7]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huagui.stl"),
                                                                expand_radius=.001)
        self.rgt.lnks[7]['rgba'] = [.57, .57, .57, 1]
        self.rgt.lnks[8]['name'] = "finger3"
        self.rgt.lnks[8]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "gripper1.stl"),
                                                          expand_radius=.001)
        self.rgt.lnks[8]['rgba'] = [.2, .2, .2, 1]
        self.rgt.lnks[9]['name'] = "huakuai3"
        self.rgt.lnks[9]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huakuai1.stl"),
                                                          expand_radius=.001)
        self.rgt.lnks[9]['rgba'] = [.6, .6, .6, 1]
        self.rgt.lnks[10]['name'] = "finger4"
        self.rgt.lnks[10]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.lnks[10]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "gripper2.stl"),
                                                           expand_radius=.001)
        self.rgt.lnks[10]['rgba'] = [.2, .2, .2, 1]
        self.rgt.lnks[11]['name'] = "huakuai4"
        self.rgt.lnks[11]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huakuai2.stl"),
                                                           expand_radius=.001)
        self.rgt.lnks[11]['rgba'] = [.6, .6, .6, 1]


        # reinitialize
        self.lft.reinitialize()
        self.lft.reinitialize()
        self.rgt.reinitialize()
        # jaw width
        self.jawwidth_rng = [0.0, .06]
        # jaw center
        self.jaw_center_pos = np.array([0, 0, .133])


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
        if self.lft.jnts[1]['motion_rng'][0] <= -motion_val <= self.lft.jnts[1]['motion_rng'][1]:
            self.lft.jnts[1]['motion_val'] = motion_val
            self.rgt.jnts[1]['motion_val'] = self.lft.jnts[1]['motion_val']
            self.lft.fk()
            self.rgt.fk()
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def jaw_to(self, jaw_width):
        if jaw_width > self.jawwidth_rng[1]:
            raise ValueError("The jaw_width parameter is out of range!")
        self.fk(motion_val=-jaw_width / 2.0)

    def get_jawwidth(self):
        return -self.lft.jnts[1]['motion_val'] * 2

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='lite6wrs_gripper_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.body.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                 tcp_loc_pos=tcp_loc_pos,
                                 tcp_loc_rotmat=tcp_loc_rotmat,
                                 toggle_tcpcs=False,
                                 toggle_jntscs=toggle_jntscs,
                                 toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
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
                      name='reconfgripper'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.body.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                rgba=rgba).attach_to(meshmodel)
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
    grpr = maingripper()
    # grpr.gen_meshmodel().attach_to(base)
    grpr.jaw_to(.06)
    grpr.gen_meshmodel().attach_to(base)
    base.run()