import os
import math
import numpy as np
import modeling.model_collection as mc
import modeling.collision_model as cm
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp
import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xc1
import robot_sim.end_effectors.gripper.xc330gripper2.xc330gripper2 as xc2
import robot_sim.end_effectors.gripper.PGC_300_60_W_S.PGC as PGC


class reconfgripper(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='convex_hull', name='maingripper',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']

        # main gripper
        self.body = PGC.PGC(pos=cpl_end_pos, rotmat=cpl_end_rotmat,  name='body')

        # lft gripper
        self.lft = xc1.xc330gripper(pos=self.body.lft.jnts[2]['gl_posq'],
                                    rotmat=self.body.lft.jnts[2]['gl_rotmatq'],
                                    name='lft')

        # rgt gripper
        self.rgt = xc2.xc330gripper(pos=self.body.rgt.jnts[2]['gl_posq'],
                                    rotmat=self.body.rgt.jnts[2]['gl_rotmatq'],
                                    name='rgt')

        # jaw width
        self.jawwidth_rng = [0, .063]

        self.gripper_dict = {}
        self.gripper_dict['lft'] = self.lft
        self.gripper_dict['rgt'] = self.rgt
        self.gripper_dict['main'] = self.body


    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.body.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, component_name='main', motion_val='main'):
        """
        """
        # def update_oih(component_name='arm'):
        #     for obj_info in self.oih_infos:
        #         gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
        #         obj_info['gl_pos'] = gl_pos
        #         obj_info['gl_rotmat'] = gl_rotmat

        def update_component(motion_val):
            status= self.gripper_dict[component_name].fk(motion_val=motion_val/2)
            # status_lft = self.gipper_dict['lft'].fk(motion_val=motion_val)
            # status_rgt = self.gipper_dict['rgt'].fk(motion_val=motion_val)
            self.gripper_dict['lft'].fix_to(
                pos=self.gripper_dict['main'].lft.jnts[-1]['gl_posq'],
                rotmat=self.gripper_dict['main'].lft.jnts[-1]['gl_rotmatq'])
            self.gripper_dict['rgt'].fix_to(
                pos=self.gripper_dict['main'].rgt.jnts[-1]['gl_posq'],
                rotmat=self.gripper_dict['main'].rgt.jnts[-1]['gl_rotmatq'])
            # update_oih(component_name=component_name)
            return status

        return update_component(motion_val)

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
        # self.body.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
        #                          tcp_loc_pos=tcp_loc_pos,
        #                          tcp_loc_rotmat=tcp_loc_rotmat,
        #                          toggle_tcpcs=False,
        #                          toggle_jntscs=toggle_jntscs,
        #                          toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
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
                      name='xc330gripper'):
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
        self.lft.jaw_to(.03)


    def lg_close(self):
        '''
        Close left gripper
        '''
        self.lft.jaw_to(0)


    def rg_open(self):
        '''
        Open right gripper
        '''
        self.rgt.jaw_to(.03)


    def rg_close(self):
        '''
        Close right gripper
        '''
        self.rgt.jaw_to(0)

    def lg_jaw_to(self, jaw_width):
        '''
        Right gripper jaws to "jaw_width"
        '''
        self.lft.jaw_to(jaw_width)

    def rg_jaw_to(self, jaw_width):
        '''
        left gripper jaws to "jaw_width"
        '''
        self.rgt.jaw_to(jaw_width)

    def mg_open(self):
        '''
        Open left gripper
        '''
        self.body.jaw_to(.063)
        self.fk("main", 0.063)


    def mg_close(self):
        '''
        Close left gripper
        '''
        self.body.jaw_to(0)


    def mg_jaw_to(self, jaw_width):
        '''
        Open right gripper
        '''
        self.body.jaw_to(jaw_width)
        self.fk("main", jaw_width)

    def get_jaw_center_pos(self):
        '''
        Get jaw center position
        '''
        pass

    def get_jawwidth(self, g = "m"):
        if g == "m":
            return self.body.get_jawwidth()
        elif g =="l":
            return self.lft.get_jawwidth()
        else:
            return self.rgt.get_jawwidth()


    def center_pos_global(self, contact_center_pos, contact_center_rotmat, mg_jawwidth, g="l"):
        if g == "l":
            x = contact_center_pos + contact_center_rotmat.dot([-0.054 - mg_jawwidth, 0, 0])
            y = contact_center_pos + contact_center_rotmat.dot([-0.027 - mg_jawwidth / 2, .018, -0.2012])
            print(f"左夹爪中心在全局坐标系下的坐标为：{contact_center_pos}")
            print(f"右夹爪中心在全局坐标系下的坐标为：{x}")
            print(f"大夹爪中心在全局坐标系下的坐标为：{y}")
            return x, y
        elif g == "r":
            x = contact_center_pos + contact_center_rotmat.dot([0.054 + mg_jawwidth, 0, 0])
            y = contact_center_pos + contact_center_rotmat.dot([0.027 + mg_jawwidth / 2, -.018, -0.2012])
            print(f"左夹爪中心在全局坐标系下的坐标为：{x}")
            print(f"右夹爪中心在全局坐标系下的坐标为：{contact_center_pos}")
            print(f"大夹爪中心在全局坐标系下的坐标为：{y}")
            return x, y

    def gripper_bool(self,g='m'):
        if g == 'm':
            open_bool = self.get_jawwidth("m") > 0  # 检测大夹爪是否打开
            print(f"大夹爪是否打开：{open_bool}")
        elif g == 'l':
            open_bool = self.get_jawwidth("l") > 0   # 检测左夹爪是否打开
            print(f"左夹爪是否打开：{open_bool}")
        else:
            open_bool = self.get_jawwidth("r") > 0  # 检测右夹爪是否打开
            print(f"右夹爪是否打开：{open_bool}")


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    gm.gen_frame().attach_to(base)
    grpr = reconfgripper()
    # grpr.body.set_finger_type('a', coupling_offset_pos=np.zeros(3))
    grpr.mg_jaw_to(0.063)
    grpr.lg_close()
    grpr.rg_close()
    m_jawwidth = grpr.get_jawwidth(g='m')
    m_rotmat = grpr.rotmat
    grpr.gen_meshmodel().attach_to(base)
    base.run()


