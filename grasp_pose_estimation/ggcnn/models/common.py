'''
Description: 
Author: wangdx
Date: 2021-11-28 11:30:06
LastEditTime: 2021-11-28 12:51:02
'''
import torch
from skimage.filters import gaussian
from utils.data.structure.grasp import GRASP_WIDTH_MAX


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Qut of GG-CNN
    :param sin_img: sin outp output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    # 见论文图3下第一段
    q_img = q_img.cpu().numpy().squeeze()
    cos_img = cos_img * 2 - 1
    sin_img = sin_img * 2 - 1
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * GRASP_WIDTH_MAX
    # 采用高斯核输出最后结果
    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img
