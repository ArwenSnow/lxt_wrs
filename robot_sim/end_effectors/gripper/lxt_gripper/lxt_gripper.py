import os
import math
import numpy as np
import modeling.model_collection as mc
import modeling.collision_model as cm
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp
import drivers.devices.dynamixel_sdk.sdk_wrapper as mw
import time
import math
import numpy as np


class Gripper(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='convex_hull', name='lite6_wrs_gripper2',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # gripper base
        self.body = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(7), name='base')
        self.body.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.body.jnts[2]['loc_pos'] = np.array([0, 0, 0.003])
        self.body.jnts[3]['loc_pos'] = np.array([-0.0015, -0.0066, 0.00505])
        self.body.jnts[4]['loc_pos'] = np.array([0, 0.0132, 0])
        # self.body.jnts[5]['loc_pos'] = np.array([.0015, -.0066, -.03105])
        # self.body.jnts[6]['loc_pos'] = np.array([-.0264, -.01635, .0027])

        self.body.lnks[0]['name'] = "dianji"
        self.body.lnks[0]['loc_pos'] = np.zeros(3)
        self.body.lnks[0]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "dianji.stl"),
                                                                 expand_radius=.001)
        self.body.lnks[0]['rgba'] = [0.2, 0.2, 0.2, 1]

        self.body.lnks[1]['name'] = "base"
        self.body.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.body.lnks[1]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "base.stl"),
                                                                 expand_radius=.001)
        self.body.lnks[1]['rgba'] = [.37, .37, .37, 1]
        self.body.lnks[2]['name'] = "chilun"
        self.body.lnks[2]['loc_pos'] = np.array([0, 0, 0])
        self.body.lnks[2]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "chilun.stl"),
                                                                 expand_radius=.001)
        self.body.lnks[2]['rgba'] = [.37, .37, .37, 1]
        self.body.lnks[3]['name'] = "huagui1"
        self.body.lnks[3]['loc_pos'] = np.array([0, 0, 0])
        self.body.lnks[3]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huagui.stl"),
                                                                 expand_radius=.001)
        self.body.lnks[3]['rgba'] = [.57, .57, .57, 1]
        self.body.lnks[4]['name'] = "huagui2"
        self.body.lnks[4]['loc_pos'] = np.array([0, 0, 0])
        self.body.lnks[4]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "huagui.stl"),
                                                                 expand_radius=.001)
        self.body.lnks[4]['rgba'] = [.57, .57, .57, 1]
        # self.body.lnks[5]['name'] = "gripper_base"
        # self.body.lnks[5]['loc_pos'] = np.array([0, 0, 0])
        # self.body.lnks[5]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "gripper_base.stl"),
        #                                                          expand_radius=.001)
        # self.body.lnks[5]['rgba'] = [.57, .57, .57, 1]
        # self.body.lnks[6]['name'] = "zhijianzujian"
        # self.body.lnks[6]['loc_pos'] = np.array([0, 0, 0])
        # self.body.lnks[6]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "zhijian.stl"),
        #                                                          expand_radius=.001)
        # self.body.lnks[6]['rgba'] = [.57, .57, .57, 1]

        # lft finger
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(2), name='lft_finger')
        self.lft.jnts[1]['loc_pos'] = np.array([0.01064, 0.0106, 0.01389])
        self.lft.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0,0)
        self.lft.jnts[1]['type'] = 'prismatic'
        self.lft.jnts[1]['motion_rng'] = [.0, .025]
        self.lft.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft.jnts[2]['loc_pos'] = np.array([0, 0, 0])
        self.lft.jnts[2]['loc_motionax'] = np.array([1, 0, 0])

        self.lft.lnks[1]['name'] = "finger1"
        self.lft.lnks[1]['mesh_file'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "gripper1.stl"), cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._finger_cdnp, expand_radius=.001)
        self.lft.lnks[1]['rgba'] = [.2, .2, .2, 1]
        self.lft.lnks[2]['name'] = "huakuai1"
        self.lft.lnks[2]['mesh_file'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "huakuai1.stl"), cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._finger_cdnp, expand_radius=.001)
        self.lft.lnks[2]['rgba'] = [.6, .6, .6, 1]

        # rgt finger
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(2), name='rgt_finger')
        self.rgt.jnts[1]['loc_pos'] = np.array([-.01536,-.0106,.01389])
        self.rgt.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.rgt.jnts[1]['type'] = 'prismatic'
        self.rgt.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt.jnts[2]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])

        self.rgt.lnks[1]['name'] = "finger2"
        self.rgt.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.rgt.lnks[1]['mesh_file'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "gripper2.stl"), cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._finger_cdnp, expand_radius=.001)
        self.rgt.lnks[1]['rgba'] = [.2, .2, .2, 1]
        self.rgt.lnks[2]['name'] = "huakuai2"
        self.rgt.lnks[2]['mesh_file'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "huakuai2.stl"), cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._finger_cdnp, expand_radius=.001)
        self.rgt.lnks[2]['rgba'] = [.6, .6, .6, 1]

        # reinitialize
        self.body.reinitialize(cdmesh_type=cdmesh_type)
        self.lft.reinitialize(cdmesh_type=cdmesh_type)
        self.rgt.reinitialize(cdmesh_type=cdmesh_type)
        # jaw width
        self.jawwidth_rng = [0.0, .03]
        # jaw center
        self.jaw_center_pos = np.array([0, 0, .3])
        # collision detection
        self.all_cdelements = []
        # self.enable_cc(toggle_cdprimit=enable_cc)


    @staticmethod
    def _finger_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.0035, 0.004, .025 + .003),
                                              x=.0035 + radius, y=0.0032 + radius, z=.025 + .003 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(.008, 0.028 - .002, -.011),
                                              x=.018 + radius, y=0.008 + radius, z=.011 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(-.005, 0.012 - .002, -.002 + .0025),
                                              x=.005 + radius, y=0.008 + radius, z=.002 + .0025 + radius)
        collision_node.addSolid(collision_primitive_c2)
        return collision_node

    @staticmethod
    def _hnd_base_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, 0, .031),
                                              x=.036 + radius, y=0.038 + radius, z=.031 + radius)
        collision_node.addSolid(collision_primitive_c0)  # 0.62
        collision_primitive_c1 = CollisionBox(Point3(0, 0, .067),
                                              x=.036 + radius, y=0.027 + radius, z=.003 + radius)
        collision_node.addSolid(collision_primitive_c1)  # 0.06700000
        #
        collision_primitive_c2 = CollisionBox(Point3(.006, .049, .0485),
                                              x=.02 + radius, y=.02 + radius, z=.015 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_c3 = CollisionBox(Point3(0, 0, .08),
                                              x=.013 + radius, y=0.013 + radius, z=.005 + radius)
        collision_node.addSolid(collision_primitive_c3)

        return collision_node

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.body, [0, 1])
            self.cc.add_cdlnks(self.lft, [1])
            self.cc.add_cdlnks(self.rgt, [1])
            activelist = [self.body.lnks[0],
                          self.body.lnks[1],
                          self.lft.lnks[1],
                          self.rgt.lnks[1]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements
        else:
            self.all_cdelements = [self.body.lnks[0],
                                   self.body.lnks[1],
                                   self.lft.lnks[1],
                                   self.rgt.lnks[1]]
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
        self.body.fix_to(cpl_end_pos, cpl_end_rotmat)
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
        jaw_width = 0.03-jaw_width
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
                      name='lxt_gripper'):
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

    def lg_open(self):
        '''
        Open left gripper
        '''
        pass

    def lg_close(self):
        '''
        Close left gripper
        '''
        pass

    def rg_open(self):
        '''
        Open right gripper
        '''
        pass

    def rg_close(self):
        '''
        Close right gripper
        '''
        pass

    def rg_jaw_to(self, jawwidth):
        '''
        Right gripper jaws to "jawwidth"
        '''
        pass

    def lg_jaw_to(self, jawwidth):
        '''
        left gripper jaws to "jawwidth"
        '''
        pass

    def mg_open(self):
        '''
        Main gripper open
        '''
        pass

    def mg_close(self):
        '''
        Main gripper open
        '''
        pass

    def mg_jaw_to(self, jawwidth):
        '''
        Main gripper jaws to "jawwidth"
        '''
        pass

# def move_line(wide, id_group):
#       encoder=34000*wide+700
#       encoder = int(np.round(encoder))
#       print(encoder)
#       return encoder
#
# peripheral_baud = 57600  #设置波特率为57600
# com = 'COM3'  #设置通信接口为com3
# dxl_con = mw.DynamixelMotor(com, baud_rate=peripheral_baud, toggle_group_sync_write=True)
# dxl_con.set_dxl_op_mode(5, 1)  #设置id为1的电机操作模式为5
#
# dxl_con.enable_dxl_torque(1)  #启用电机扭矩
# print(dxl_con.get_dxl_pos(1))  #获取电机位置
# print(dxl_con.get_dxl_vel(1))  #获取电机速度
#
# dxl_con.set_dxl_goal_pos(tgt_pos=700, dxl_id=1)  #将id为1的电机目标位置设为~
# a=dxl_con.get_dxl_pos(1)
# print(a)
# dxl_con.set_dxl_goal_pos(2000,0)
# time.sleep(2)
# dxl_con.set_dxl_goal_pos(2000,1)
# time.sleep(2)
# dxl_con.set_dxl_goal_pos(2000,2)
# time.sleep(4)
# print('//')
# print(dxl_con.get_dxl_pos(1))
# a=dxl_con.get_dxl_pos(1)
# print(a)
#
# time.sleep(1)
# time.sleep(1)

if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import robot_sim.end_effectors.gripper.lite6_wrs_gripper as gp1

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    gm.gen_frame().attach_to(base)
    # cm.CollisionModel("meshes/dual_realsense.stl", expand_radius=.001).attach_to(base)
    grpr = Gripper(enable_cc=True)
    # grpr.gen_meshmodel().attach_to(base)
    grpr.jaw_to(.0)
    grpr.gen_meshmodel().attach_to(base)
    # grpr.show_cdprimit()
    base.run()

