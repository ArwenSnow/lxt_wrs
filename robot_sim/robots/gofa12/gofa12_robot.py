import os
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.gofa12.gofa12 as rbt
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as hnd
import robot_sim.robots.robot_interface as ri
import visualization.panda.world as wd
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim.manipulators.machinetool.machinetool_gripper as machine
import basis.robot_math as rm
import transforms3d as t3d

class GOFA12(ri.RobotInterface):  #定义机器人类：GOFA12，继承自ri.RobotInterface类

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="gofa12", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # base plate
        self.base_stand = jl.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     homeconf=np.zeros(0),
                                     name='base_stand')          #创建一个base_stand属性，是一个jl.JLchain的实例

        self.base_stand.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "base_plate.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)  # 为底座设置碰撞模型
        self.base_stand.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand.reinitialize()
        # arm
        arm_homeconf = np.zeros(6)
        self.arm = rbt.GOFA12(pos=pos,     #创建一个arm属性，是一个GOFA12类的实例
                            rotmat=self.base_stand.jnts[-1]['gl_rotmatq'],
                            homeconf=arm_homeconf,
                            name='arm', enable_cc=False)
        # gripper
        self.hnd = hnd.Robotiq85(pos=self.arm.jnts[-1 ]['gl_posq'],   #创建一个hnd属性，是一个Robotiq85类的实例
                                   rotmat=np.dot(rm.rotmat_from_axangle([0, 1, 0], math.pi * 1 /2), self.arm.jnts[-1 ]['gl_rotmatq']),
                            name='hnd_s', enable_cc=False)

        # tool center point
        self.arm.jlc.tcp_jnt_id = -1    #设置末端执行器的位置和姿态
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.jaw_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []     #创建oih_infos属性，具体信息有待添加
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
        collision_primitive_c0 = CollisionBox(Point3(0.0, 0.0, .1),
                                              x=.112 + radius, y=.112 + radius, z=.2 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node  #创建碰撞盒模型

    def enable_cc(self):  #碰撞检测
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.base_stand, [0])
        self.cc.add_cdlnks(self.arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.lft_outer, [0, 1, 2, 3])   #？为什么是[0, 1, 2, 3]和[1, 2, 3]
        self.cc.add_cdlnks(self.hnd.rgt_outer, [1, 2, 3])
        activelist = [self.base_stand.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.lft_outer.lnks[0],
                      self.hnd.lft_outer.lnks[1],
                      self.hnd.lft_outer.lnks[2],
                      self.hnd.lft_outer.lnks[3],
                      self.hnd.rgt_outer.lnks[1],
                      self.hnd.rgt_outer.lnks[2],
                      self.hnd.rgt_outer.lnks[3]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.base_stand.lnks[0],
                    self.arm.lnks[1]]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_outer.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.lft_outer.lnks[3],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[3]]
        self.cc.set_cdpair(fromlist, intolist)
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold(objcm)  #？确保机器人手持物体的碰撞模型与其他物体进行正确的碰撞检测

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_stand.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_stand.jnts[-1]['gl_posq'], rotmat=self.base_stand.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # self.machine.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat


    def fk(self, component_name='arm', jnt_values=np.zeros(6)):     #定义fk方法
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):   #定义了update_oih 的方法
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
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
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

    def jaw_to(self, hnd_name='hnd_s', jawwidth=0.0):
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
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
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
                       name='xarm7_shuidi_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_stand.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
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
        self.machine.gen_stickmodel(toggle_tcpcs=False,
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
                      name='xarm_shuidi_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        if is_robot:
            self.base_stand.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
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
        if is_machine:
            self.machine.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel

    def door_to(self, door_width):
        self.machine.door_to(door_width)

    def chunck_to(self, chunck_width):
        self.machine.jaw_to(chunck_width)


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])
    gm.gen_frame().attach_to(base)

    robot_s = GOFA12(enable_cc=True)
    robot_s.jaw_to(.02)   #执行机器人夹爪的动作，将夹爪移动到指定的位置（0.02）
    robot_s.gen_meshmodel(toggle_tcpcs=False, toggle_jntscs=False).attach_to(base)
    #生成机器人的三维模型并将其附加到基本场景 base 上。toggle_tcpcs 和 toggle_jntscs 参数控制是否显示工具中心点坐标系和关节坐标系。
    tgt_pos = np.array([.25, .2, .15]) #定义目标位置
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3) #创建一个旋转矩阵 tgt_rotmat，该旋转矩阵表示绕 Y 轴旋转 π/3 弧度的旋转。
    # gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    robot_s.show_cdprimit() #显示机器人的碰撞模型
    # robot_s.gen_stickmodel().attach_to(base)
    base.run()
    component_name = 'arm'
    jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    robot_s.fk(component_name, jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=False)
    robot_s_meshmodel.attach_to(base)
    # robot_s.show_cdprimit()
    robot_s.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = robot_s.is_collided()
    toc = time.time()
    print(result, toc - tic)
    base.run()