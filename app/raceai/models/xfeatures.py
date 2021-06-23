#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file xfeatures.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-12-11 20:33

import cv2
import numpy as np

from raceai.utils.misc import race_load_class


class FlannMatcher(cv2.FlannBasedMatcher):
    def __init__(self, cfg):
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        if 'index_params' in cfg:
            index_params = dict(cfg['index_params'])
        if 'search_params' in cfg:
            search_params = dict(cfg['search_params'])
        super().__init__(index_params, search_params)


class Features2dSift(object):
    def __init__(self, cfg):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.matcher = race_load_class(cfg.matcher.class_name)(cfg.matcher.params)

    def extract(self, icon):
        return self.sift.detectAndCompute(icon, None)

    def matches(self, des1, des2):
        matches = self.matcher.knnMatch(des1, des2, k=2)
        top_results1 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                top_results1.append([m])

        matches = self.matcher.knnMatch(des2, des1, k=2)
        top_results2 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                top_results2.append([m])

        top_results = []
        for match1 in top_results1:
            match1QueryIndex = match1[0].queryIdx
            match1TrainIndex = match1[0].trainIdx

            for match2 in top_results2:
                match2QueryIndex = match2[0].queryIdx
                match2TrainIndex = match2[0].trainIdx

                if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                    top_results.append(match1)
        return top_results


class PlateColorFeatures(object):
    def __init__(self, cfg):
        ellsize, iterations, edge_thickness = 15, 3, 30
        if 'ellsize' in cfg:
            ellsize = cfg['ellsize']
        if 'iterations' in cfg:
            iterations = cfg['iterations']
        if 'edge_thickness' in cfg:
            edge_thickness = cfg['edge_thickness']

        self.ellsize = ellsize
        self.iterations = iterations
        self.edge_thickness = edge_thickness
        colors_names = ['black', 'gray', 'white', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
        self.name2index = {name: x for x, name in enumerate(colors_names)}

    def hsv_color(self, h, s, v):
        if s > 43 and v > 46:
            if (0 <= h and h <= 10) or (156 <= h and h <= 180):
                return 'red'
            if 11 <= h and h <= 25:
                return 'orange'
            if 26 <= h and h <= 34:
                return 'yellow'
            if 35 <= h and h <= 77:
                return 'green'
            if 78 <= h and h <= 99:
                return 'cyan'
            if 100 <= h and h <= 124:
                return 'blue'
            if 125 <= h and h <= 155:
                return 'purple'
        else:
            if s < 43 and v <= 220:
                return 'gray'
            if s < 30 and v >= 221:
                return 'white'
        return 'black'

    def extract_features(self, path):
        img_bgr = cv2.imread(path)
        img_bgr = cv2.copyMakeBorder(
            img_bgr,
            top=2,
            bottom=2,
            left=2,
            right=2,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255])

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img_gray)
        # img_blur = cv2.GaussianBlur(img_gray, (15, 15), 0)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

        img_canny = cv2.Canny(img_blur, 30, 150)

        ellipses = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.ellsize, self.ellsize))

        mask_close = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, ellipses, iterations=self.iterations)

        contours = cv2.findContours(
            mask_close,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)[0]

        contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        mask = np.zeros(img_gray.shape, np.uint8)

        cv2.drawContours(mask, [contours_sorted[0]], 0, (255, 255, 255), self.edge_thickness)

        contours = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)[0]

        contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        mask = np.zeros(img_gray.shape, np.uint8)
        cv2.drawContours(mask, [contours_sorted[-1]], 0, (255, 255, 255), -1)
        masked = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        img_hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        counts = [0 for x in range(len(self.name2index))]
        for item in contours_sorted[-1]:
            x, y = item[0]
            h, s, v = img_hsv[y, x, :]
            counts[self.name2index[self.hsv_color(h, s, v)]] += 1
        return counts, masked
