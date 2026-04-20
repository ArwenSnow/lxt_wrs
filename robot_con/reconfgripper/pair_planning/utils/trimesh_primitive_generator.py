import numpy as np
import trimesh.creation

import utils.robot_math as rm
import shapely.geometry as shpg
import trimesh.primitives as tp


def gen_section(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), height_vec=np.array([0, 0, 1]), height=0.01,
                angle=30, section=8):
    pos = spos
    direction = rm.unit_vector(epos - spos)
    length = np.linalg.norm(epos - spos)
    height = height
    if np.allclose(height, 0):
        rotmat_goal = np.eye(3)
    else:
        rotmat_goal = rm.rotmat_from_two_axis(direction, rm.unit_vector(height_vec), "xz")
    rotmat = rotmat_goal
    homomat = rm.homomat_from_posrot(pos, rotmat)
    center_offset = - (rotmat[:, 2] * height / 2)
    center_offset_homo = rm.homomat_from_posrot(center_offset, np.eye(3))
    homomat = center_offset_homo.dot(homomat)
    angle_rad = np.deg2rad(angle)
    direction_boundary = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), -angle_rad / 2), [1, 0, 0])
    curve_pnts = [(np.array([0, 0, 0]) + np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), i * angle_rad / section),
                                                direction_boundary) * length)[:2] for i in range(section + 1)]
    curve_pnts.append(np.array([0, 0]))
    extrude_polygon = shpg.Polygon(curve_pnts)
    extrude_transform = homomat
    extrude_height = height
    return tp.Extrusion(polygon=extrude_polygon, transform=extrude_transform, height=extrude_height)


def gen_cylinder(radius=0.01, height=0.1, section=100, homomat=np.eye(4)):
    return tp.Cylinder(radius=radius, height=height, sections=section, transform=homomat)


def gen_cone(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), radius=0.005, sections=8):
    height = np.linalg.norm(spos - epos)
    pos = spos
    rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)
    homomat = rm.homomat_from_posrot(pos, rotmat)
    cone = trimesh.creation.cone(radius=radius, height=height, sections=sections, transform=homomat)
    return cone

