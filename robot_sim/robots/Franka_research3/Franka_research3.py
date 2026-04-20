import os,sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))#根目录错误时
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.Franka_research3.Franka_research3 as rbt
# import robot_sim.end_effectors.gripper.frank_research3.frank_research3 as hnd
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as hnd
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3
# import robot_sim.manipulators.machinetool.machinetool_gripper as machine
import basis.robot_math as rm
import motion.probabilistic.rrt_connect as rrtc
from typing import Literal
from trac_ik import TracIK


class Franka_research3(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="Franka_research3", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # base plate,机器人组件初始化
        self.base_stand = jl.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     homeconf=np.zeros(0),
                                     name='base_stand')
        self.base_stand.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "table.STL")
        self.base_stand.lnks[0]['collision_model'] = cm.CollisionModel(
            self.base_stand.lnks[0]['meshfile'],
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.base_stand.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand.reinitialize()

        # arm机械臂初始化
        arm_homeconf = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, math.pi/4*3])
        self.arm = rbt.Franka(pos=self.base_stand.jnts[-1]['gl_posq']+np.array([0, 0.2, 0]),
                              rotmat=self.base_stand.jnts[-1]['gl_rotmatq']@rm.rotmat_from_axangle([0, 0, 1], math.pi/2),
                              homeconf=arm_homeconf,
                              name='arm', enable_cc=False)

        # gripper手爪初始化，固定于机械臂末端
        self.hnd = hnd.Reconfgripper(pos=self.arm.jnts[-1]['gl_posq'],
                                     rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                     name='hnd_s', enable_cc=False)

        # tool center point工具中心点
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.hnd.lft.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.lft.jaw_center_rotmat

        self.iksolver_cache = {}

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

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, 0.6, -0.375),
                                              x=0.66 + radius, y=0.6 + radius, z=0.375 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.base_stand, [0])
        self.cc.add_cdlnks(self.arm, [1, 2, 3, 4, 5, 6, 7])
        self.cc.add_cdlnks(self.hnd.body.lft, [0, 1])
        self.cc.add_cdlnks(self.hnd.body.rgt, [1])
        self.cc.add_cdlnks(self.hnd.lft.lft, [0, 1])
        self.cc.add_cdlnks(self.hnd.lft.rgt, [1])
        self.cc.add_cdlnks(self.hnd.rgt.lft, [0, 1])
        self.cc.add_cdlnks(self.hnd.rgt.rgt, [1])

        activelist = [self.base_stand.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.arm.lnks[7],

                      self.hnd.body.lft.lnks[0],
                      self.hnd.body.lft.lnks[1],
                      self.hnd.body.rgt.lnks[1],
                      self.hnd.lft.lft.lnks[0],
                      self.hnd.lft.lft.lnks[1],
                      self.hnd.lft.rgt.lnks[1],
                      self.hnd.rgt.lft.lnks[0],
                      self.hnd.rgt.lft.lnks[1],
                      self.hnd.rgt.rgt.lnks[1]]

        self.cc.set_active_cdlnks(activelist)
        for oih_info in self.oih_infos:        # 抓住物体后，添加碰撞体
            objcm = oih_info['collision_model']
            self.hold(objcm=objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_stand.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_stand.jnts[-1]['gl_posq'] + rotmat @ np.array([0, 0.2, 0]),
                        rotmat=self.base_stand.jnts[-1]['gl_rotmatq'] @ rm.rotmat_from_axangle([0, 0, 1], math.pi/2))
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'],
                        rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def get_tgt_pose_in_rbt(self, tgt_pos, tgt_rotmat):
        arm_pos = self.arm.pos
        arm_rot = self.arm.rotmat
        wd_to_rbt = rm.homomat_from_posrot(arm_pos, arm_rot)

        hand_pos = self.hnd.jaw_center_pos + np.array([0, 0, 0.05665])
        hand_rot = (self.hnd.jaw_center_rotmat @ rm.rotmat_from_axangle([0, 0, 1], -math.pi/4)
                    @ rm.rotmat_from_axangle([0, 0, 1], math.pi))

        end_to_hand = rm.homomat_from_posrot(hand_pos.dot(hand_rot), hand_rot)
        hand_to_end = np.linalg.inv(end_to_hand)

        tgt_homomat_wd = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
        end_homomat_wd = tgt_homomat_wd.dot(hand_to_end)
        end_homomat_rbt = np.linalg.inv(wd_to_rbt).dot(end_homomat_wd)
        new_tgt_rot = end_homomat_rbt[:3, :3]
        new_tgt_pos = end_homomat_rbt[:3, 3]
        return new_tgt_pos, new_tgt_rot

    def tracik(self,
               urdf_path: str = "C:/Users/11154/Documents/GitHub/lxt_wrs/robot_sim/robots/Franka_research3/urdf/franka.urdf",
               base_link_name: str = 'Franka_research3_link_0',
               tip_link_name: str = 'Franka_research3_link_7',
               tgt_pos=np.zeros(3),
               tgt_rotmat=np.eye(3),
               seed_jnt_values=None,
               solver_type: Literal['Speed', 'Distance', 'Manip1', 'Manip2'] = "Distance"):
        new_tgt_pos, new_tgt_rot = self.get_tgt_pose_in_rbt(tgt_pos, tgt_rotmat)
        # gm.gen_frame(new_tgt_pos, new_tgt_rot).attach_to(base)

        key = (urdf_path, base_link_name, tip_link_name, solver_type)
        if key not in self.iksolver_cache:
            self.iksolver_cache[key] = TracIK(base_link_name=base_link_name,
                                              tip_link_name=tip_link_name,
                                              urdf_path=urdf_path,
                                              solver_type=solver_type)
        iksolver = self.iksolver_cache[key]
        seed_jnt_values = seed_jnt_values if seed_jnt_values is not None else np.zeros(7)
        return iksolver.ik(new_tgt_pos, new_tgt_rot, seed_jnt_values)

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

    def jaw_to(self, hnd_name='hnd_s', jawwidth=0.0):   # 手爪开到jawwidth宽度
        self.hnd.jaw_to(jawwidth)

    def hold(self, hnd_name, objcm, jawwidth=None):    # 抓取物体并将其添加到碰撞检测系统
        """
        the objcm is added as a part of the robot_s to the cd checker
        输入：
        jawwidth: 宽度
        objcm:  抓住的物体
        输出：
        rel_pos:
        rel_rotmat:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].lg_jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.base_stand.lnks[0],
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
        释放物体并从碰撞检测系统中移除
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].lg_jaw_to(0)
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
                       name='xarm7_shuidi_mobile_stickmodel'):
        # 生成机器人的杆状模型（用于简化显示），手爪上也会生成杆状模型

        stickmodel = mc.ModelCollection(name=name)
        # self.base_stand.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
        #                                tcp_loc_pos=tcp_loc_pos,
        #                                tcp_loc_rotmat=tcp_loc_rotmat,
        #                                toggle_tcpcs=False,
        #                                toggle_jntscs=toggle_jntscs,
        #                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        # self.hnd.gen_stickmodel(toggle_tcpcs=False,
        #                         toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      is_robot=True,
                      is_machine=None,
                      name='xarm_shuidi_mobile_meshmodel'):
        # 生成机器人的网格模型（用于真实感显示）
        meshmodel = mc.ModelCollection(name=name)
        if is_robot:
            self.base_stand.gen_meshmodel(
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat,
                                          toggle_tcpcs=False,
                                          toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            self.arm.gen_meshmodel(
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
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[-2, 4, 1.5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    robot_s = Franka_research3(enable_cc=True)
    robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=False).attach_to(base)
    robot_s.show_cdprimit()

    robot_s.fix_to(np.array([2, 0, 0]), rm.rotmat_from_axangle([0, 0, 1], math.pi/2))
    robot_s.gen_meshmodel().attach_to(base)

    # test_pos, test_rot = robot_s.get_gl_tcp("arm")
    # tgt_pos = test_pos.copy()
    # tgt_pos[0] += .1
    # tgt_pos[1] += .1
    # tgt_pos[2] -= .1
    # tgt_rotmat = test_rot
    #
    # jnt_values = robot_s.ik("arm", tgt_pos, tgt_rotmat)
    # robot_s.fk("arm", jnt_values=jnt_values)
    # robot_s_meshmodel = robot_s.gen_meshmodel(rgba=(0, 0, 1, 1), toggle_tcpcs=False)
    # robot_s_meshmodel.attach_to(base)
    #
    # urdf_file = "C:/Users/11154/Documents/GitHub/lxt_wrs/robot_sim/robots/Franka_research3/urdf/franka.urdf"
    # gm.gen_frame(tgt_pos, tgt_rotmat).attach_to(base)
    # seed_jnt = robot_s.get_jnt_values("arm")
    # time_1 = time.perf_counter()
    # conf = robot_s.tracik(urdf_file, 'Franka_research3_link_0', 'Franka_research3_link_7', tgt_pos, tgt_rotmat, seed_jnt)
    # time_2 = time.perf_counter()
    # print("tracik求解时间：", time_2 - time_1)
    # robot_s.fk('arm', conf)
    # robot_s.gen_meshmodel(rgba=(0, 1, 0, 1), ).attach_to(base)

    base.run()
