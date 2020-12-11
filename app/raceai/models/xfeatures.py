#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file xfeatures.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-12-11 20:33

import cv2

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
