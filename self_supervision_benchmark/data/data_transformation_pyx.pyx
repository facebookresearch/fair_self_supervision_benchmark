cimport cython
cimport numpy as np
import numpy as np
import cv2
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
def convert2Lab(np.ndarray[np.uint8_t, ndim=3, mode='c'] img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_lab[:, :, 0] = (img_lab[:, :, 0] * (100.0 / 255.0)) - 50.0
    img_lab[:, :, 1:] = img_lab[:, :, 1:] - 128.0
    return img_lab


# Image should be in format CHW
@cython.boundscheck(False)
@cython.wraparound(False)
def batch_color_normalization(np.ndarray[float, ndim=4, mode='c'] imgs,
                              np.ndarray[float, ndim=1, mode='c'] mean,
                              np.ndarray[float, ndim=1, mode='c'] stddev):
    imgs = imgs - mean[np.newaxis, :, np.newaxis, np.newaxis]
    imgs = imgs / stddev[np.newaxis, :, np.newaxis, np.newaxis]
    return imgs


# Image should be in format CHW
@cython.boundscheck(False)
@cython.wraparound(False)
def batch_mean_normalization(np.ndarray[float, ndim=4, mode='c'] imgs,
                              np.ndarray[float, ndim=1, mode='c'] mean):
    imgs = imgs - mean[np.newaxis, :, np.newaxis, np.newaxis]
    return imgs


# Image should be in format CHW
@cython.boundscheck(False)
@cython.wraparound(False)
def batch_lighting(np.ndarray[float, ndim=4, mode='c'] imgs,
                   np.ndarray[float, ndim=1, mode='c'] rgb):
    for i in range(len(imgs)):
        for idx in range(imgs[i].shape[0]):
            imgs[i][idx] = imgs[i][idx] + rgb[2 - idx]
    return imgs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :, :] scalar_mult(
          float[:, :, :] image,
          float alpha) nogil:
    cdef int I = image.shape[0]
    cdef int J = image.shape[1]
    cdef int K = image.shape[2]
    cdef int i
    cdef float* image_p = &image[0, 0, 0]
    for i in range(I * J * K):
        image_p[i] = image_p[i] * alpha
    return image


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :] scalar_mult2d(
          float[:, :] image,
          float alpha) nogil:
    cdef int I = image.shape[0]
    cdef int J = image.shape[1]
    cdef int i
    cdef float* image_p = &image[0, 0]
    for i in range(I * J):
        image_p[i] = image_p[i] * alpha
    return image


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :, :] fill(
          float[:, :, :] image,
          float alpha) nogil:
    cdef int I = image.shape[0]
    cdef int J = image.shape[1]
    cdef int K = image.shape[2]
    cdef float* image_p = &image[0, 0, 0]
    cdef int i
    for i in range(I * J * K):
        image_p[i] = alpha
    return image


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum3d(
        float[:, :, :] image) nogil:
    cdef float total = 0
    cdef int I = image.shape[0]
    cdef int J = image.shape[1]
    cdef int K = image.shape[2]
    cdef float* image_p = &image[0, 0, 0]
    cdef int i
    for i in range(I * J * K):
        total += image_p[i]
    return total

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum2d(
        float[:, :] image) nogil:
    cdef float total = 0
    cdef int I = image.shape[0]
    cdef int J = image.shape[1]
    cdef float* image_p = &image[0, 0]
    cdef int i
    for i in range(I * J):
        total += image_p[i]
    return total


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :, :] add_left(
          float[:, :, :] dst,
          float[:, :, :] src) nogil:
    cdef int I = dst.shape[0]
    cdef int J = dst.shape[1]
    cdef int K = dst.shape[2]
    cdef float* dst_p = &dst[0, 0, 0]
    cdef float* src_p = &src[0, 0, 0]
    cdef int i
    for i in range(I * J * K):
        dst_p[i] = dst_p[i] + src_p[i]
    return dst


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :] add_left2d(
          float[:, :] dst,
          float[:, :] src) nogil:
    cdef int I = dst.shape[0]
    cdef int J = dst.shape[1]
    cdef float* dst_p = &dst[0, 0]
    cdef float* src_p = &src[0, 0]
    cdef int i
    for i in range(I * J):
        dst_p[i] = dst_p[i] + src_p[i]
    return dst


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :, :] blend(
          float[:, :, :] image1,
          float[:, :, :] image2,
          float alpha) nogil:
    return add_left(scalar_mult(image1, alpha),
                    scalar_mult(image2, (1 - alpha)))


# image should be in format CHW and the channels in order BGR
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :, :] grayscale(float[:, :, :] img_gray,
                              float[:, :, :] image) nogil:
    # R -> 0.299, G -> 0.587, B -> 0.114
    scalar_mult2d(img_gray[2], 0.299)
    scalar_mult2d(img_gray[1], 0.587)
    scalar_mult2d(img_gray[0], 0.114)
    add_left2d(img_gray[2], img_gray[1])
    add_left2d(img_gray[2], img_gray[0])
    img_gray[1] = img_gray[2]
    img_gray[0] = img_gray[2]
    return img_gray


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :, :] saturation(
        float alpha,
        float[:, :, :] image,
        float[:, :, :] img_gray) nogil:
    grayscale(img_gray, image)
    return blend(image, img_gray, alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :, :] brightness(
        float alpha,
        float[:, :, :] image) nogil:
    return scalar_mult(image, alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :, :] contrast(
        float alpha,
        float[:, :, :] image,
        float[:, :, :] img_gray) nogil:
    grayscale(img_gray, image)
    cdef float s = sum2d(img_gray[0])
    cdef int div_s = img_gray.shape[1] * img_gray.shape[2]
    s = s / div_s
    fill(img_gray, s)
    return blend(image, img_gray, alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
def batch_color_jitter(np.ndarray[float, ndim=4, mode='c'] images_np,
                       np.ndarray[int, ndim=2, mode='c'] orders_np,
                       np.ndarray[float, ndim=2, mode='c'] alphas_np):
    cdef float[:, :, :, :] images = images_np
    cdef float[:, :, :, :] img_gray = images.copy()
    cdef int[:, :] orders = orders_np
    cdef float[:, :] alphas = alphas_np
    cdef int I = images_np.shape[0]
    cdef int IDX = orders_np.shape[1]
    cdef int i, idx
    with nogil:
        for i in prange(I):
            for idx in range(IDX):
                if orders[i][idx] == 0:
                    images[i] = brightness(alphas[i][0], images[i])
                if orders[i][idx] == 1:
                    images[i] = contrast(alphas[i][1], images[i], img_gray[i])
                if orders[i][idx] == 2:
                    images[i] = saturation(alphas[i][2], images[i], img_gray[i])
    return images_np
