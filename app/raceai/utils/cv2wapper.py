#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file cv2wapper.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-12-11 17:37

import numpy as np
import cv2
import matplotlib.pylab as plt


class CV2Image(object):
    def __init__(self):
        pass

    @staticmethod
    def open(img_path):
        print(img_path)
        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def resize(img, size):
        return cv2.resize(img, size)

    @staticmethod
    def rotate(img, angle, scale=1.0, center=None):
        (h, w) = img.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(img, M, (w, h))

    @staticmethod
    def gaussian_blur(img, ksize=(11, 11)):
        return cv2.GaussianBlur(img, ksize, 0)

    @staticmethod
    def median_blur(img, ksize=5):
        return cv2.medianBlur(img, ksize)

    @staticmethod
    def contrast(img, clip_limit=3.0, tile_grid_size=(8,8)):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_rgb(img, alpha=True):
        if alpha:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_thresh(img, threshold=190, reverse=False):
        if reverse:
            return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]

    @staticmethod
    def to_canny(img, thresh1=75, thresh2=200):
        return cv2.Canny(img, thresh1, thresh2)

    @staticmethod
    def show(img, figsize=(8, 8), gray=False, isbgr=True):
        plt.figure(figsize=figsize)
        plt.axis('off')
        if isbgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        if gray:
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            plt.imshow(img)


class CV2Contour(object):
    def __init__(self):
        pass

    @staticmethod
    def make_white_background(img, contour):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask_black = np.zeros(gray.shape, np.uint8)
        mask_bg_black_fg_white = cv2.drawContours(mask_black, [contour], -1, (255, 255, 255), cv2.FILLED)
        fg_masked = cv2.bitwise_and(img, img, mask=mask_bg_black_fg_white)

        mask_bg_white_fg_black = cv2.bitwise_not(mask_bg_black_fg_white)
        mask_white = np.full(img.shape, 255, np.uint8)
        bg_masked = cv2.bitwise_and(mask_white, mask_white, mask=mask_bg_white_fg_black)

        return cv2.bitwise_or(fg_masked, bg_masked)

    @staticmethod
    def make_out_roi(img, contour, square=False):
        x, y, w, h = cv2.boundingRect(contour)
        if square:
            if w < h:
                x += int((w - h) / 2)
                w = h
            else:
                y += int((h - w) / 2)
                h = w
        return img[y:y + h, x:x + w], [x, y, w, h]

    @staticmethod
    def on_draw(img, contour, color=(255, 0, 0), thickness=2, rect=False):
        if rect:
            x, y, w, h = cv2.boundingRect(contour)
            out_image = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), color=color, thickness=thickness)
        else:
            out_image = cv2.drawContours(img.copy(), [contour], -1, color=color, thickness=thickness)
        return out_image

    def grab_by_area(img, threshold=None, reverse=False, all=False, area=False):
        if threshold is not None:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)[1]
            if reverse: # cv2.THRESH_BINARY_INV
                img_thresh = cv2.bitwise_not(img_thresh)
        else:
            img_thresh = img.copy()
        if all:
            cnts, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            cnts, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if area:
            cnts = [c for c in cnts if cv2.contourArea(c) > area]
        return sorted(cnts, key=cv2.contourArea, reverse=True)
