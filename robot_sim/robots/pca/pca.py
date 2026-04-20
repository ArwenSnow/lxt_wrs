import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim._kinematics.jlchain_ik as jlik
import robot_sim.manipulators.pca.pca as rbt
import robot_sim.end_effectors.gripper.pca_handle.pca_handle as hnd
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3


class Pca(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="pca", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # baseplate
        self.base_3 = jl.JLChain(pos=pos,
                                 rotmat=rotmat,
                                 homeconf=np.zeros(0),
                                 name='base_3')
        self.base_3.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "table.STL"),
            cdprimit_type="box", expand_radius=.01)
        self.base_3.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_3.reinitialize()

        base_1_pos = pos + [0, -0.18, -648/1000]
        self.base_1 = jl.JLChain(pos=self.base_3.jnts[-1]['gl_posq'] + [0, -0.18, -0.648],
                                 rotmat=self.base_3.jnts[-1]['gl_rotmatq'],
                                 homeconf=np.zeros(0),
                                 name='base_1')
        self.base_1.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "base_1.STL"),
            cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._base_1_combined_cdnp)
        self.base_1.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_1.reinitialize()

        self.base_2 = jl.JLChain(pos=self.base_3.jnts[-1]['gl_posq'] + [0, -0.155, 0.302],
                                 rotmat=self.base_3.jnts[-1]['gl_rotmatq'],
                                 homeconf=np.zeros(0),
                                 name='base_2')
        self.base_2.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "base_2.STL"),
            cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._base_2_combined_cdnp)
        self.base_2.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_2.reinitialize()

        # arm
        arm_homeconf = np.zeros(7)
        arm_y = rm.rotmat_from_axangle([0, 1, 0], -math.pi/2)
        arm_z = rm.rotmat_from_axangle([0, 0, 1], -math.pi/2)
        self.arm = rbt.Pca(pos=self.base_3.jnts[-1]['gl_posq'] + [0.18, -0.13611, 0.6492],
                           rotmat=self.base_3.jnts[-1]['gl_rotmatq']@arm_z@arm_y,
                           homeconf=arm_homeconf,
                           name='arm', enable_cc=False)

        # gripper
        self.hnd = hnd.PcaHandle(pos=self.arm.jnts[-1]['gl_posq'],
                                 rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                 name='hnd', enable_cc=False)

        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.jaw_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd

        # self.ik_solver = jlik.JLChainIK(jlc_object=1)

    @staticmethod
    def _base_1_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, 0, .495),
                                              x=.025 + radius, y=.025 + radius, z=.495 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    @staticmethod
    def _base_2_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, .041, .1461),
                                              x=.04 + radius, y=.041 + radius, z=.1861 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, .07799, .3412),
                                              x=.22 + radius, y=.003 + radius, z=.056 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.base_1, [0])
        self.cc.add_cdlnks(self.base_2, [0])
        self.cc.add_cdlnks(self.base_3, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6, 7])
        self.cc.add_cdlnks(self.hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.hnd.rgt, [1])
        activelist = [self.base_1.lnks[0],
                      self.base_2.lnks[0],
                      self.base_3.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.lft.lnks[0],
                      self.hnd.lft.lnks[1],
                      self.hnd.rgt.lnks[1]
                      ]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.base_1.lnks[0],
                    self.base_2.lnks[0],
                    self.arm.lnks[1],
                    self.base_3.lnks[0]]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft.lnks[0],
                    self.hnd.lft.lnks[1],
                    self.hnd.rgt.lnks[1]
                    ]
        self.cc.set_cdpair(fromlist, intolist)
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_3.fix_to(pos=pos, rotmat=rotmat)
        self.base_1.fix_to(pos=self.base_3.jnts[-1]['gl_posq'] + rotmat@[0, -0.18, -648/1000],
                           rotmat=self.base_3.jnts[-1]['gl_rotmatq'])
        self.base_2.fix_to(pos=self.base_3.jnts[-1]['gl_posq'] + rotmat@[0, -0.155, 0.302],
                           rotmat=self.base_3.jnts[-1]['gl_rotmatq'])
        arm_y = rm.rotmat_from_axangle([0, 1, 0], -math.pi/2)
        arm_z = rm.rotmat_from_axangle([0, 0, 1], -math.pi/2)
        self.arm.fix_to(pos=self.base_3.jnts[-1]['gl_posq'] + rotmat@[0.18, -0.13611, 0.6492],
                        rotmat=self.base_3.jnts[-1]['gl_rotmatq']@arm_z@arm_y)
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])

        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def tracik(self,
               component_name: str = "arm",
               urdf_path: str = os.path.join(os.path.dirname(__file__), "urdf/gofa5.urdf"),
               base_link_name: str = 'base_link',
               tip_link_name: str = 'link_6',
               tgt_pos=np.zeros(3),
               tgt_rotmat=np.eye(3),
               seed_jnt_values=None):
        arm_pos = np.array([0.13747, -0.03748, 0.015])
        arm_rot = np.eye(3)
        wd_to_rbt = rm.homomat_from_posrot(arm_pos, arm_rot)
        hand_pos = np.array([0., 0., 0.2035])
        hand_rot = rm.rotmat_from_axangle([0, 0, 1], np.pi).dot(rm.rotmat_from_axangle([0, 1, 0], -np.pi/2))
        end_to_hand = rm.homomat_from_posrot(hand_pos.dot(hand_rot), hand_rot)
        hand_to_end = np.linalg.inv(end_to_hand)
        tgt_homomat_wd = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
        end_homomat_wd = tgt_homomat_wd.dot(hand_to_end)
        end_homomat_rbt = np.linalg.inv(wd_to_rbt).dot(end_homomat_wd)
        new_tgt_rot = end_homomat_rbt[:3, :3]
        new_tgt_pos = end_homomat_rbt[:3, 3]

        return self.manipulator_dict[component_name].tracik(urdf_path=urdf_path,
                                                            base_link_name=base_link_name,
                                                            tip_link_name=tip_link_name,
                                                            tgt_pos=new_tgt_pos,
                                                            tgt_rotmat=new_tgt_rot,
                                                            seed_jnt_values=seed_jnt_values)

    def fk(self, component_name='arm', jnt_values=np.zeros(7)):
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
        else:
            raise ValueError("The given component name is not supported!")

    def ik(self,
           component_name: str = "arm",
           tgt_pos=np.zeros(3),
           tgt_rotmat=np.eye(3),
           seed_jnt_values=None,
           max_niter=200,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima: str = "end",
           toggle_debug=False):
        return self.manipulator_dict[component_name].ik(tgt_pos,
                                                        tgt_rotmat,
                                                        seed_jnt_values=seed_jnt_values,
                                                        max_niter=max_niter,
                                                        tcp_jnt_id=tcp_jnt_id,
                                                        tcp_loc_pos=tcp_loc_pos,
                                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                                        local_minima=local_minima,
                                                        toggle_debug=toggle_debug)

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self, hand_name, jawwidth=0.0):
        self.hnd.jaw_to(jawwidth)

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
        intolist = [self.base_3.lnks[0],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4]]
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
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='pca_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_1.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.base_2.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.base_3.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcpcs=False,
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
                      name='pca_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        if is_robot:
            self.base_1.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            self.base_2.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            self.base_3.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
            self.hnd.gen_meshmodel(toggle_tcpcs=False,
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

    base = wd.World(cam_pos=[-1, 2, 1], lookat_pos=[0, 0, 0])
    # gm.gen_frame().attach_to(base)

    robot_s = Pca(enable_cc=True)
    robot_s.gen_meshmodel(toggle_tcpcs=False, toggle_jntscs=False).attach_to(base)
    pos, rotmat = robot_s.get_gl_tcp("arm")
    gm.gen_frame(pos, rotmat).attach_to(base)
    print("pos=", pos)
    print("rotmat=", rotmat)
    base.run()
