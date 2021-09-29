#!/bin/python

import cv2
import numpy as np
import matplotlib.pyplot as plt


def rotation(image, angle):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angle, scale=1)

    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    return cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)


def resize(image, scale):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(scale * w), int(scale * h)))


def blend_image_iter(bg_files, fg_files, fg_labels, tilesizes=[30], imgcount=100):
    bg_images = [cv2.imread(f) for f in bg_files]
    fg_images = [cv2.imread(f) for f in fg_files]

    tileposes = [
        [
            [
                (
                    int(0.5 * (img.shape[1] % tilesize)) + tilesize * x,
                    int(0.5 * (img.shape[0] % tilesize)) + tilesize * y
                )
                for x in range(int(img.shape[1] / tilesize)) for y in range(int(img.shape[0] / tilesize))
            ]
            for tilesize in tilesizes
        ]
        for img in bg_images
    ]

    for _ in range(imgcount):
        # bg
        choice = random.choice(range(len(bg_images)))
        bg_new = bg_images[choice].copy()
        if random.random() > 0.2:
            ts_choice = random.choice(range(len(tilesizes)))
            tilesize = tilesizes[ts_choice]
            tilepos_0 = tileposes[choice][ts_choice]
            tilepos_1 = tilepos_0.copy()
            random.shuffle(tilepos_1)
            for (x0, y0), (x1, y1) in zip(tilepos_0, tilepos_1):
                bg_new[y0:y0 + tilesize, x0:x0 + tilesize] = bg_images[choice][y1:y1 + tilesize, x1:x1 + tilesize]
            bg_new = cv2.medianBlur(bg_new, ksize=7)

        # fg
        labels = []
        annotations = []
        # for choice in random.choices(range(len(fg_images)), k=random.choice([1, 1, 1, 2])):
        for choice in random.choices(range(len(fg_images)), k=random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])):
            fgimg = fg_images[choice]
            fgimg = resize(fgimg, 0.9 + 1.1 * random.random())
            if random.random() > 0.8:
                fgimg = rotation(fgimg, -5 + 10 * random.random())
            fgx = random.randint(0, bg_new.shape[1] - fgimg.shape[1])
            fgy = random.randint(0, bg_new.shape[0] - fgimg.shape[0])
            bg_new[fgy:fgy + fgimg.shape[0], fgx:fgx + fgimg.shape[1]] = fgimg
            labels.append('%d %.6f %.6f %.6f %.6f' %(
                fg_labels[choice],
                (fgx + int(0.5 * fgimg.shape[1])) / bg_new.shape[1],
                (fgy + int(0.5 * fgimg.shape[0])) / bg_new.shape[0],
                fgimg.shape[1] / bg_new.shape[1], fgimg.shape[0] / bg_new.shape[0]
            ))
        bg_new = cv2.medianBlur(bg_new, ksize=3)

        yield bg_new, labels



iter_blend = blend_image_iter(
    ['./bg_incise.png', './bg_stiletto.png'],
    ['./incise.png', './incise_1.png', './incise_2.png', './stiletto.png', './stiletto_1.png', './stiletto_2.png'],
    [0, 0, 0, 1, 1, 1],
    tilesizes=[100], imgcount=20)
