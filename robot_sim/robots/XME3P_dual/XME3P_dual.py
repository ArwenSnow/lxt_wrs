import os
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.XME3p.xme3p as rbt
import robot_sim.end_effectors.gripper.dh76.dh76 as hnd
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3
import time
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import copy
from trac_ik import TracIK
from typing import Literal


class XME3P_dual(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="gofa5", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # base plate
        self.base_stand2 = jl.JLChain(pos=pos+[0.0875, 0, 0],
                                      rotmat=rotmat,
                                      homeconf=np.zeros(0),
                                      name='base_stand2')

        self.base_stand2.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "80160.stl"),
            cdprimit_type="box", expand_radius=-.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.base_stand2.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand2.reinitialize()
        self.base_connect2 = jl.JLChain(pos=pos+[0.0875, 0, 0.8],
                                        rotmat=rotmat,
                                        homeconf=np.zeros(0),
                                        name='base_stand2')

        self.base_connect2.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "connectarm.stl"),
            cdprimit_type="box", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.base_connect2.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_connect2.reinitialize()

        # arm
        arm_homeconf = np.array([0.0, -math.pi/12, 0.0, math.pi/6, 0.0, math.pi/12, 0.0])
        self.lft_arm = rbt.XME3P(pos=pos + [0.0875, -0.18818, 0.72697],
                                 rotmat=np.dot(rm.rotmat_from_axangle(axis=np.array([1, 0, 0]), angle=math.pi/4*3),
                                               rm.rotmat_from_axangle(axis=np.array([0, 1, 0]), angle=math.pi/4)),
                                 homeconf=arm_homeconf,
                                 name='lft_arm', enable_cc=True)
        # gripper
        self.lft_hnd = hnd.Dh76(pos=self.lft_arm.jnts[-1]['gl_posq'],
                                rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'],
                                name='lft_hnd', enable_cc=False)

        self.rgt_arm = rbt.XME3P(pos=pos + [0.0875, 0.18818, 0.72697],
                                 rotmat=np.dot(rm.rotmat_from_axangle(axis=np.array([1, 0, 0]), angle=-math.pi/4*3),
                                               rm.rotmat_from_axangle(axis=np.array([0, 1, 0]), angle=math.pi/4)),
                                 homeconf=arm_homeconf,
                                 name='rgt_arm', enable_cc=True)
        # gripper
        self.rgt_hnd = hnd.Dh76(pos=self.rgt_arm.jnts[-1]['gl_posq'],
                                rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'],
                                name='rgt_hnd', enable_cc=False)

        # tool center point
        self.lft_arm.jlc.tcp_jnt_id = -1
        self.lft_arm.jlc.tcp_loc_pos = self.lft_hnd.jaw_center_pos
        self.lft_arm.jlc.tcp_loc_rotmat = self.lft_hnd.jaw_center_rotmat

        self.rgt_arm.jlc.tcp_jnt_id = -1
        self.rgt_arm.jlc.tcp_loc_pos = self.rgt_hnd.jaw_center_pos
        self.rgt_arm.jlc.tcp_loc_rotmat = self.rgt_hnd.jaw_center_rotmat

        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['lft_arm'] = self.lft_arm
        self.manipulator_dict['lft_hnd'] = self.lft_arm
        self.hnd_dict['lft_hnd'] = self.lft_hnd
        self.hnd_dict['lft_arm'] = self.lft_hnd
        self.manipulator_dict['rgt_arm'] = self.rgt_arm
        self.manipulator_dict['rgt_hnd'] = self.rgt_arm
        self.hnd_dict['rgt_hnd'] = self.rgt_hnd
        self.hnd_dict['rgt_arm'] = self.rgt_hnd
        # self.iksolver_cache = {}

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-0.1, 0.0, 0.14 - 0.82),
                                              x=.35 + radius, y=.3 + radius, z=.14 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, -.3),
                                              x=.112 + radius, y=.112 + radius, z=.3 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.lft_arm, [0, 1, 2, 3, 4, 5, 6, 7])
        self.cc.add_cdlnks(self.lft_hnd.lft, [0,1])
        self.cc.add_cdlnks(self.lft_hnd.rgt,[1])
        self.cc.add_cdlnks(self.rgt_arm, [0, 1, 2, 3, 4, 5, 6, 7])
        self.cc.add_cdlnks(self.rgt_hnd.lft,[0,1])
        self.cc.add_cdlnks(self.rgt_hnd.rgt,[1])
        self.cc.add_cdlnks(self.base_stand2, [0])
        activelist = [self.lft_arm.lnks[1],
                      self.lft_arm.lnks[2],
                      self.lft_arm.lnks[3],
                      self.lft_arm.lnks[4],
                      self.lft_arm.lnks[5],
                      self.lft_arm.lnks[6],
                      self.lft_arm.lnks[7],
                      self.rgt_arm.lnks[1],
                      self.rgt_arm.lnks[2],
                      self.rgt_arm.lnks[3],
                      self.rgt_arm.lnks[4],
                      self.rgt_arm.lnks[5],
                      self.rgt_arm.lnks[6],
                      self.rgt_arm.lnks[7],
                      self.lft_hnd.lft.lnks[0],
                      self.lft_hnd.lft.lnks[1],
                      self.lft_hnd.rgt.lnks[1],
                      self.rgt_hnd.lft.lnks[0],
                      self.rgt_hnd.lft.lnks[1],
                      self.rgt_hnd.rgt.lnks[1],
                      self.base_stand2.lnks[0]
                      ]

        self.cc.set_active_cdlnks(activelist)

        fromlist = [self.lft_arm.lnks[1],
                    self.rgt_arm.lnks[1],
                    self.base_stand2.lnks[0]
                    ]
        intolist = [self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self.lft_hnd.lft.lnks[0],
                    self.lft_hnd.lft.lnks[1],
                    self.lft_hnd.rgt.lnks[1],

                    self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self.rgt_hnd.lft.lnks[0],
                    self.rgt_hnd.lft.lnks[1],
                    self.rgt_hnd.rgt.lnks[1]
                    ]

        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.lft_arm.lnks[1],
                    self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self.lft_hnd.lft.lnks[0],
                    self.lft_hnd.lft.lnks[1],
                    self.lft_hnd.rgt.lnks[1],
                    ]
        intolist = [self.rgt_arm.lnks[1],
                    self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self.rgt_hnd.lft.lnks[0],
                    self.rgt_hnd.lft.lnks[1],
                    self.rgt_hnd.rgt.lnks[1]
                    ]

        self.cc.set_cdpair(fromlist, intolist)
    #
    #
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.lft_arm.fix_to(pos=self.base_stand2.jnts[-1]['gl_posq'], rotmat=self.base_stand2.jnts[-1]['gl_rotmatq'])
        self.lft_hnd.fix_to(pos=self.lft_arm.jnts[-1]['gl_posq'], rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'])

        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.lft_arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    # def jaw_center_pos(self):
    #     return self.machine.jaw_center_pos

    # def jaw_center_rot(self):
    #     return self.machine.jaw_center_rot

    def get_tgt_pose_in_rbt(self, tgt_pos, tgt_rotmat):
        arm_pos = self.lft_arm.pos
        arm_rot = self.lft_arm.rotmat
        wd_to_rbt = rm.homomat_from_posrot(arm_pos, arm_rot)
        hand_pos = np.array([0., 0., 0.2035])  # 手爪与机器人末端的位置与旋转，根据使用手爪进行改变
        hand_rot = np.eye(3)
        end_to_hand = rm.homomat_from_posrot(hand_pos.dot(hand_rot), hand_rot)
        hand_to_end = np.linalg.inv(end_to_hand)
        tgt_homomat_wd = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
        end_homomat_wd = tgt_homomat_wd.dot(hand_to_end)
        end_homomat_rbt = np.linalg.inv(wd_to_rbt).dot(end_homomat_wd)
        new_tgt_rot = end_homomat_rbt[:3, :3]
        new_tgt_pos = end_homomat_rbt[:3, 3]
        return new_tgt_pos, new_tgt_rot


    def fk(self, component_name='arm', jnt_values=np.zeros(6)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 7:
                raise ValueError("An 1x7 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)
        if component_name =='both':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 14:
                raise ValueError("An 1x14 npdarray must be specified to move the arm!")
            return update_component('lft_arm', jnt_values[0:7]), \
                   update_component('rgt_arm', jnt_values[7:14])
        else:
            raise ValueError("The given component name is not supported!")

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    # def get_jnt_init(self, component_name):
    #     if component_name in self.manipulator_dict:
    #         return self.arm.init_jnts
    #     else:
    #         raise ValueError("The given component name is not supported!")

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self,hand_name, jawwidth=0.0, both_jawwidth=[0,0]):
        if hand_name == 'lft_hnd':
            self.lft_hnd.jaw_to(jawwidth)
            self.lft_arm.jlc.tcp_loc_pos = self.lft_hnd.jaw_center_pos
            self.lft_arm.jlc.tcp_loc_rotmat = self.lft_hnd.jaw_center_rotmat
        if hand_name == 'rgt_hnd':
            self.rgt_hnd.jaw_to(jawwidth)
            self.rgt_arm.jlc.tcp_loc_pos = self.rgt_hnd.jaw_center_pos
            self.rgt_arm.jlc.tcp_loc_rotmat = self.rgt_hnd.jaw_center_rotmat
        if hand_name == 'both':
            self.lft_hnd.jaw_to(both_jawwidth[0])
            self.rgt_hnd.jaw_to(both_jawwidth[1])
            self.lft_arm.jlc.tcp_loc_pos = self.lft_hnd.jaw_center_pos
            self.lft_arm.jlc.tcp_loc_rotmat = self.lft_hnd.jaw_center_rotmat
            self.rgt_arm.jlc.tcp_loc_pos = self.rgt_hnd.jaw_center_pos
            self.rgt_arm.jlc.tcp_loc_rotmat = self.rgt_hnd.jaw_center_rotmat


    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.lft_arm.lnks[1],
                    self.lft_arm.lnks[2],
                    self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4]]
        # intolist = [self.lft_arm.lnks[1]]
        # self.lft_arm.gen_meshmodel().attach_to(base)
        # self.lft_arm.show_cdprimit()
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def grasp(self, obj_cmodel):
        try:
            self.cc.add_cdlnks(obj_cmodel, [0])  # 假设物体只有一个链接
        except ValueError as e:
            if "already added" not in str(e):
                raise e

        # 更新激活列表，包含被抓取的物体
        current_activelist = self.cc.get_active_cdlnks()  # 获取当前激活列表
        current_activelist.append(obj_cmodel.lnks[0])  # 添加物体链接
        self.cc.set_active_cdlnks(current_activelist)  # 设置新的激活列表

        # 存储物体信息
        self.oih_infos.append({
            'obj_cmodel': obj_cmodel,
            'gl_pos': obj_cmodel.get_pos(),
            'gl_rotmat': obj_cmodel.get_rotmat()
        })


    def release(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            print(jawwidth)
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                # self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='XME3P_dual_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_stand2.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                        tcp_loc_pos=tcp_loc_pos,
                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                        toggle_tcpcs=False,
                                        toggle_jntscs=toggle_jntscs,
                                        toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.base_connect2.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                        tcp_loc_pos=tcp_loc_pos,
                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                        toggle_tcpcs=False,
                                        toggle_jntscs=toggle_jntscs,
                                        toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)

        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      is_machine=None,
                      is_robot=True,
                      name='XME3P_dual_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        if is_robot:
            self.base_stand2.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                           tcp_loc_pos=tcp_loc_pos,
                                           tcp_loc_rotmat=tcp_loc_rotmat,
                                           toggle_tcpcs=False,
                                           toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            self.base_connect2.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                           tcp_loc_pos=tcp_loc_pos,
                                           tcp_loc_rotmat=tcp_loc_rotmat,
                                           toggle_tcpcs=False,
                                           toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            self.lft_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
            self.lft_hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
            self.rgt_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
            self.rgt_hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    print(np.dot(rm.rotmat_from_axangle(axis=np.array([1,0,0]),angle=math.pi/4*3),rm.rotmat_from_axangle(axis=np.array([0,1,0]),angle=math.pi/4)))
    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])
    # base.run()
    table = cm.CollisionModel(
                "meshes/wholetable.stl",
                cdprimit_type="box", expand_radius=-.003)
    table.set_rgba([0.35, 0.35, 0.35, 1])
    table.attach_to(base)
    gm.gen_frame().attach_to(base)
    # base.run()
    robot_s = XME3P_dual(enable_cc=True)
    robot_s.gen_meshmodel().attach_to(base)

    a = robot_s.lft_arm.get_jnt_values()
    print("左臂关节角度 = ", a)
    b = robot_s.rgt_arm.get_jnt_values()
    print("右臂关节角度 = ", b)
    # base.run()
    # xx = robot_s.ik(component_name='lft_arm',tgt_pos=np.array([0.4,0.3,0.1]),tgt_rotmat=rm.rotmat_from_axangle([1,0,0],math.pi))
    # print(xx)
    # robot_s.fk(component_name='lft_arm',jnt_values=xx)
    # robot_s.gen_meshmodel().attach_to(base)
    # xx = robot_s.ik(component_name='lft_arm',tgt_pos=np.array([0.3,-0.8,0.1]),tgt_rotmat=rm.rotmat_from_axangle([1,0,0],math.pi))
    # print(xx)
    # robot_s.fk(component_name='lft_arm',jnt_values=xx)
    # robot_s.gen_meshmodel().attach_to(base)
    #
    # xx = robot_s.ik(component_name='lft_arm',tgt_pos=np.array([0.9,0,0.1]),tgt_rotmat=rm.rotmat_from_axangle([1,0,0],math.pi))
    # print(xx)
    # robot_s.fk(component_name='lft_arm',jnt_values=xx)
    # robot_s.gen_meshmodel().attach_to(base)
    #
    # xx = robot_s.ik(component_name='lft_arm',tgt_pos=np.array([0.9,-0.6,0.1]),tgt_rotmat=rm.rotmat_from_axangle([1,0,0],math.pi))
    # print(xx)
    # robot_s.fk(component_name='lft_arm',jnt_values=xx)
    # robot_s.gen_meshmodel().attach_to(base)
    # gm.gen_stick(np.array([0.4,0.3,0.1]),np.array([0.3,-0.8,0.1])).attach_to(base)
    # gm.gen_stick(np.array([0.9,0,0.1]),np.array([0.9,-0.6,0.1])).attach_to(base)
    # gm.gen_stick(np.array([0.4,0.3,0.1]),np.array([0.9,0,0.1])).attach_to(base)
    # gm.gen_stick(np.array([0.3,-0.8,0.1]),np.array([0.9,-0.6,0.1])).attach_to(base)

    robot_s.fk(component_name='lft_arm',jnt_values=np.array([-46.365/180*math.pi,6.431/180*math.pi,-12.610/180*math.pi,42.037/180*math.pi,7.905/180*math.pi,67.354/180*math.pi,-17.065/180*math.pi]))
    robot_s.gen_meshmodel().attach_to(base)
    base.run()
    xx = robot_s.ik(component_name='lft_arm',tgt_pos=np.array([0.4,0.,0.1]),tgt_rotmat=rm.rotmat_from_axangle([1,0,0],-math.pi/2))
    print(xx)
    robot_s.fk(component_name='lft_arm',jnt_values=xx)
    robot_s.gen_meshmodel().attach_to(base)
    xx = robot_s.ik(component_name='lft_arm',tgt_pos=np.array([0.4,0.,0.3]),tgt_rotmat=rm.rotmat_from_axangle([1,0,0],-math.pi/2))
    print(xx)
    robot_s.fk(component_name='lft_arm',jnt_values=xx)
    robot_s.gen_meshmodel().attach_to(base)

    xx = robot_s.ik(component_name='lft_arm',tgt_pos=np.array([0.9,0.,0.1]),tgt_rotmat=rm.rotmat_from_axangle([1,0,0],-math.pi/2))
    print(xx)
    robot_s.fk(component_name='lft_arm',jnt_values=xx)
    robot_s.gen_meshmodel().attach_to(base)
    xx = robot_s.ik(component_name='lft_arm',tgt_pos=np.array([0.9,0.,0.7]),tgt_rotmat=rm.rotmat_from_axangle([1,0,0],-math.pi/2))
    print(xx)
    robot_s.fk(component_name='lft_arm',jnt_values=xx)
    robot_s.gen_meshmodel().attach_to(base)
    gm.gen_stick(np.array([0.4,0.,0.1]),np.array([0.4,0.,0.3])).attach_to(base)
    gm.gen_stick(np.array([0.9,0.,0.1]),np.array([0.9,0.,0.7])).attach_to(base)
    gm.gen_stick(np.array([0.4,0.,0.1]),np.array([0.9,0.,0.1])).attach_to(base)
    gm.gen_stick(np.array([0.4,0.,0.3]),np.array([0.9,0.,0.7])).attach_to(base)

    base.run()
    robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=False).attach_to(base)
    robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=False).attach_to(base)
    tgt_pos = np.array([.25, .2, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    # gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # robot_s.show_cdprimit()
    # robot_s.gen_stickmodel().attach_to(base)
    # base.run()
    # component_name = 'arm'
    # jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=False,toggle_jntscs=False)
    robot_s_meshmodel.attach_to(base)
    # robot_s.hnd.show_cdprimit()
    # robot_s.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = robot_s.is_collided()
    toc = time.time()
    print(result, toc - tic)
    base.run()
