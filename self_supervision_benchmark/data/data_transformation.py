# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This module implements various data transformation techniques on images.
Assumption: The image format is assumed to be CHW.
Implemented transformations:
1. Convert image to LAB space
2. Color normalization
3. Mean normalization
4. Pad image
5. Horizontal flip
6. Random Crop
7. Center Crop
8. Five Crop (center crop and 4 corner crops)
9. Ten Crop (standard process)
10. Ten Crop (random)
11. Scale
12. Global Resize
13. Random Scale Jittering
14. Random size crop
15. Lighting
16. Grayscale
17. Saturation
18. Brightness
19. Contrast
20. Color Jitter
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import cv2
import numpy as np
import math

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def convert2lab(img, index):
    # img is [0, 255] , HWC, BGR format, uint8 type
    assert len(img.shape) == 3, 'Image should have dim H x W x 3'
    assert img.shape[2] == 3, 'Image should have dim H x W x 3'
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # 8-bit image range -> L [0, 255], A [0, 255], B [0, 255]. Rescale it to:
    # L [0, 100], A [-128, 127], B [-128, 127]
    img_lab = img_lab.astype(np.float32)
    img_lab[:, :, 0] = (img_lab[:, :, 0] * (100.0 / 255.0)) - 50.0
    img_lab[:, :, 1:] = img_lab[:, :, 1:] - 128.0
    ############################ debugging ####################################
    # img_lab_bw = img_lab.copy()
    # img_lab_bw[:, :, 1:] = 0.0
    # img_lab_bgr = cv2.cvtColor(img_lab_bw, cv2.COLOR_Lab2BGR)
    # img_lab_bgr = img_lab_bgr.astype(np.float32)
    # img_lab_RGB = img_lab_bgr[:, :, [2, 1, 0]]        # BGR to RGB
    # img_lab_RGB = img_lab_RGB - np.min(img_lab_RGB)
    # img_lab_RGB /= np.max(img_lab_RGB) + np.finfo(np.float64).eps
    # plt.imshow(img_lab_RGB)
    # np.save('<output_path>/lab_{}.npy'.format(index), img_lab_bgr)
    ######################### debugging over ##################################
    return img_lab


# Image should be in format CHW or 10 x C x H x W (if 10-crop)
def color_normalization(img, mean, stddev, ten_crop=False):
    if ten_crop:
        assert len(mean) == img.shape[1], 'channel mean not computed properly'
        assert len(stddev) == img.shape[1], 'channel stddev not computed properly'
        img = img - mean[np.newaxis, :, np.newaxis, np.newaxis]
        img = img / stddev[np.newaxis, :, np.newaxis, np.newaxis]
    else:
        assert len(mean) == img.shape[0], 'channel mean not computed properly'
        assert len(stddev) == img.shape[0], 'channel stddev not computed properly'
        for idx in range(img.shape[0]):
            img[idx] = img[idx] - mean[idx]
            img[idx] = img[idx] / stddev[idx]
    return img


# Image should be in format CHW or 10 x C x H x W (if 10-crop)
def mean_normalization(img, mean, ten_crop=False):
    if ten_crop:
        assert len(mean) == img.shape[1], 'channel mean not computed properly'
        img = img - mean[np.newaxis, :, np.newaxis, np.newaxis]
    else:
        assert len(mean) == img.shape[0], 'channel mean not computed properly'
        img = img - mean[:, np.newaxis, np.newaxis]
    return img


def pad_image(pad_size, image, order='CHW'):
    if order == 'CHW':
        img = np.pad(
            image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode=str('constant')
        )
    elif order == 'HWC':
        img = np.pad(
            image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode=str('constant')
        )
    return img


def horizontal_flip(prob, image, order='CHW'):
    if np.random.uniform() < prob:
        if order == 'CHW':
            image = np.asarray(image).swapaxes(2, 0)
            image = image[::-1]
            image = image.swapaxes(0, 2)
        elif order == 'HWC':
            # use opencv for flipping image
            image = cv2.flip(image, 1)
    return image


# random crop from larger image with optional zero padding.
# Image can be in CHW or HWC format. Specify the image order
def random_crop(image, size, pad_size=0, order='CHW'):
    # explicitly dealing processing per image order to avoid flipping images
    if pad_size > 0:
        image = pad_image(pad_size=pad_size, image=image, order=order)
    # image format should be CHW
    if order == 'CHW':
        if image.shape[1] == size and image.shape[2] == size:
            return image
        height = image.shape[1]
        width = image.shape[2]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = image[:, y_offset:y_offset + size, x_offset:x_offset + size]
        assert cropped.shape[1] == size, "Image not cropped properly"
        assert cropped.shape[2] == size, "Image not cropped properly"
    elif order == 'HWC':
        if image.shape[0] == size and image.shape[1] == size:
            return image
        height = image.shape[0]
        width = image.shape[1]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = image[y_offset:y_offset + size, x_offset:x_offset + size, :]
        assert cropped.shape[0] == size, "Image not cropped properly"
        assert cropped.shape[1] == size, "Image not cropped properly"
    return cropped


# crop to centered rectangle. Image should be of the format HWC
def center_crop(size, image):
    # print('center_crop: {}'.format(size))
    # print('center_crop image: {}'.format(image.shape))
    height = image.shape[0]
    width = image.shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))
    cropped = image[y_offset:y_offset + size, x_offset:x_offset + size, :]
    # print('cropped: {}'.format(cropped.shape))
    assert cropped.shape[0] == size, "Image height not cropped properly"
    assert cropped.shape[1] == size, "Image width not cropped properly"
    return cropped


# image should be in format HWC
def five_crop(size, image, crop_images):
    height = image.shape[0]
    width = image.shape[1]
    # given an image, crop the 4 corners and center of given size
    center_cropped = center_crop(size, image)
    crop_images.extend([center_cropped])
    # crop the top left corner:
    crop_images.extend([image[0:size, 0:size, :]])
    # crop the top right corner
    crop_images.extend([image[0:size, width - size:width, :]])
    # crop bottom left corner
    crop_images.extend([image[height - size:height, 0:size, :]])
    # crop bottom right corner
    crop_images.extend([image[height - size:height, width - size:width, :]])
    return crop_images


# image should be in format HWC
def ten_crop(size, image):
    ten_crop_images = []

    # For the original image, crop the center and 4 corners
    ten_crop_images = five_crop(size, image, ten_crop_images)

    # Flip the image horizontally
    flipped = horizontal_flip(1.0, image, order='HWC')
    ten_crop_images = five_crop(size, flipped, ten_crop_images)

    # convert this into 10 x H x W x C
    return np.concatenate(
        [arr[np.newaxis] for arr in ten_crop_images]).astype(np.float32)


def ten_random_crop(size, image):
    ten_random_crop_images = []
    for _ in range(10):
        img = random_crop(image, size, order='HWC')
        ten_random_crop_images.append(img)
    # convert this into 10 x H x W x C
    return np.concatenate(
        [arr[np.newaxis] for arr in ten_random_crop_images]).astype(np.float32)


# Image should be in format HWC. Scale the smaller edge of image to size.
def scale(size, image):
    H, W = image.shape[0], image.shape[1]
    if ((W <= H and W == size) or (H <= W and H == size)):
        return image
    new_width, new_height = size, size
    if W < H:
        new_height = int(math.floor((float(H) / W) * size))
    else:
        new_width = int(math.floor((float(W) / H) * size))
    img = cv2.resize(image, (new_width, new_height))
    return img.astype(np.float32)


# Image should be in format HWC. Resize the entire image to the given size
# without worrying about the aspect ratio.
def global_resize(size, image, order='HWC'):
    img = cv2.resize(image, (size, size))
    return img


# ResNet style scale jittering: randomly select the scale from
# [1/max_size, 1/min_size]
def random_scale_jitter(image, min_size, max_size):
    # randomly select a scale from [1/480, 1/256] for example
    img_scale = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))
    image = scale(img_scale, image)
    return image


# Random crop with size 8% - 100% image area and aspect ratio in [3/4, 4/3]
# Reference: GoogleNet paper
# Image should be in format HWC
def random_sized_crop(image, size, min_scale=0.08, max_scale=1.0):
    for _ in range(0, 10):
        height = image.shape[0]
        width = image.shape[1]
        area = height * width
        target_area = np.random.uniform(min_scale, max_scale) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w, h = h, w
        if h <= height and w <= width:
            if height == h:
                y_offset = 0
            else:
                y_offset = np.random.randint(0, height - h)
            if width == w:
                x_offset = 0
            else:
                x_offset = np.random.randint(0, width - w)
            y_offset = int(y_offset)
            x_offset = int(x_offset)
            cropped = image[y_offset:y_offset + h, x_offset:x_offset + w, :]
            assert cropped.shape[0] == h and cropped.shape[1] == w, \
                "Wrong crop size"
            cropped = cv2.resize(cropped, (size, size))
            return cropped

    return center_crop(size, scale(size, image))


# Image should have channel order BGR and CHW format
def lighting(img, alphastd, eigval, eigvec, alpha=None):
    if alphastd == 0:
        return img
    # generate alpha1, alpha2, alpha3
    if alpha is None:
        alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eigvec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1, dtype=np.float32
    )
    for idx in range(img.shape[0]):
        img[idx] = img[idx] + rgb[2 - idx]
    return img


def blend(image1, image2, alpha):
    return image1 * alpha + image2 * (1 - alpha)


# image should be in format CHW and the channels in order BGR
def grayscale(image):
    # R -> 0.299, G -> 0.587, B -> 0.114
    img_gray = np.copy(image)
    gray_channel = 0.299 * image[2] + 0.587 * image[1] + 0.114 * image[0]
    img_gray[0] = gray_channel
    img_gray[1] = gray_channel
    img_gray[2] = gray_channel
    return img_gray


def saturation(var, image):
    img_gray = grayscale(image)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def brightness(var, image):
    img_bright = np.zeros(image.shape)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_bright, alpha)


def contrast(var, image):
    img_gray = grayscale(image)
    img_gray.fill(np.mean(img_gray[0]))
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def color_jitter(image, img_brightness=0, img_contrast=0, img_saturation=0):
    jitter = []
    if img_brightness != 0:
        jitter.append('brightness')
    if img_contrast != 0:
        jitter.append('contrast')
    if img_saturation != 0:
        jitter.append('saturation')
    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == 'brightness':
                image = brightness(img_brightness, image)
            elif jitter[order[idx]] == 'contrast':
                image = contrast(img_contrast, image)
            elif jitter[order[idx]] == 'saturation':
                image = saturation(img_saturation, image)
    return image
