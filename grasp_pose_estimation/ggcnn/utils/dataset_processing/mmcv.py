# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import math
import numpy as np


def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
        out (ndarray): The output destination.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, dst=out, interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imresize_like(img, dst_img, return_scale=False, interpolation='bilinear'):
    """Resize image to the same size of a given image.

    Args:
        img (ndarray): The input image.
        dst_img (ndarray): The target image.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Same as :func:`resize`.

    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = dst_img.shape[:2]
    return imresize(img, (w, h), return_scale, interpolation)


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imflip(img, direction='horizontal'):
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or "vertical".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    else:
        return np.flip(img, axis=0)


def imflip_(img, direction='horizontal'):
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or "vertical".

    Returns:
        ndarray: The flipped image (inplace).
    """
    assert direction in ['horizontal', 'vertical']
    if direction == 'horizontal':
        return cv2.flip(img, 1, img)
    else:
        return cv2.flip(img, 0, img)


def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
    return rotated


def bbox_clip(bboxes, img_shape):
    """Clip bboxes to fit the image shape.

    Args:
        bboxes (ndarray): Shape (..., 4*k)
        img_shape (tuple[int]): (height, width) of the image.

    Returns:
        ndarray: Clipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
    cmin[0::2] = img_shape[1] - 1
    cmin[1::2] = img_shape[0] - 1
    clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
    return clipped_bboxes


def bbox_scaling(bboxes, scale, clip_shape=None):
    """Scaling bboxes w.r.t the box center.

    Args:
        bboxes (ndarray): Shape(..., 4).
        scale (float): Scaling factor.
        clip_shape (tuple[int], optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).

    Returns:
        ndarray: Scaled bboxes.
    """
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes


def imcrop(img, bboxes, scale=1.0, pad_fill=None):
    """Crop image patches.

    3 steps: scale the bboxes -> clip bboxes -> crop and pad.

    Args:
        img (ndarray): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale (float, optional): Scale ratio of bboxes, the default value
            1.0 means no padding.
        pad_fill (Number | list[Number]): Value to be filled for padding.
            Default: None, which means no padding.

    Returns:
        list[ndarray] | ndarray: The cropped image patches.
    """
    chn = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert len(pad_fill) == chn

    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
        if pad_fill is None:
            patch = img[y1:y2 + 1, x1:x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            if chn == 1:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
            else:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, chn)
            patch = np.array(
                pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start:y_start + h, x_start:x_start + w,
                  ...] = img[y1:y1 + h, x1:x1 + w, ...]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches


def impad(img, shape, pad_val=0):
    """Pad an image to a certain shape.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w).
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas. Default: 0.

    Returns:
        ndarray: The padded image.
    """
    if not isinstance(pad_val, (int, float)):
        assert len(pad_val) == img.shape[-1]
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1], )
    assert len(shape) == len(img.shape)
    for s, img_s in zip(shape, img.shape):
        assert s >= img_s
    pad = np.empty(shape, dtype=img.dtype)
    pad[...] = pad_val
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad


def impad_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Number | Sequence[Number]): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, (pad_h, pad_w), pad_val)


def depth2Gray3(im_depth):
    """
    将深度图转至三通道8位灰度图
    (h, w, 3)
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    ret = np.expand_dims(ret, 2).repeat(3, axis=2)
    return ret

def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode='line'):
    """
    绘制grasp
    file: img路径
    grasps: list()	元素是 [row, col, angle, width]
    angle: 弧度
    width: 单位 像素
    mode: 显示模式 'line' or 'region'
    """

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        if mode == 'line':
            width = width / 2
            angle2 = calcAngle2(angle)
            k = math.tan(angle)
            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx
            if angle < math.pi:
                cv2.line(img, (col, row), (round(col + dx), round(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (round(col - dx), round(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (round(col + dx), round(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (round(col - dx), round(row + dy)), (0, 0, 255), 1)

        color_b = 0
        color_r = 0
        color_g = 255

        if mode == 'line':
            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)
        else:
            img[row, col] = [color_b, color_g, color_r]
        

    return img