#!/bin/python3

import import cv2
import numpy as np
import matplotlib.pyplot as plt

# import ipywidgets as widgets

from sklearn.cluster import KMeans
from collections import Counter


def im_read(url, rgb=True, size=None):
    if url.startswith('http'):
        response = requests.get(url)
        if response:
            imgmat = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(imgmat, cv2.IMREAD_COLOR)
        else:
            return None
    else:
        img = cv2.imread(url)

    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        if isinstance(size, int):
            size = (size, size)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def extract_blackhole_rect(imgpath, thresh, kernel, iterations, iqr):
    img_rgb = im_read(imgpath)

    # gray and threshold
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_bin = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]

    # dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel * 5, kernel))
    img_dilate = cv2.dilate(img_bin, kernel, iterations=iterations)

    # find contours
    contours = cv2.findContours(
        img_dilate,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)[0]

    # check width, height and area
    # np_data = np.array([(*cv2.boundingRect(c), cv2.contourArea(c)) for c in contours])
    np_data = np.array([cv2.boundingRect(c) for c in contours])
    np_data = np.column_stack((np_data, np_data[:, 2] * np_data[:, 3]))
    global g_data
    g_data = np_data
    Q1, medians, Q3 = np.percentile(np_data[:, 2:], [25, 50, 75], axis=0)
    IQR = Q3 - Q1
    upper_adjacent = np.clip(Q3 + IQR * iqr, Q3, np.max(np_data[:, 2:], axis=0))
    lower_adjacent = np.clip(Q1 - IQR * iqr, np.min(np_data[:, 2:], axis=0), Q1)
    outliers_mask = np.array([False] * len(np_data))

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 32));

    img = img_gray.copy()
    for c in contours:
        cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=-1)

    for i, img in enumerate([img_bin, img_dilate, img]):
        axes[0][i].imshow(img, cmap='gray')

    for i, col, title in zip((0, 1, 2), (2, 3, 4), ('Width', 'Height', 'Area')):
        axes[1][i].yaxis.grid(True)

        axes[1][i].violinplot(np_data[:, col], showmeans=True, showmedians=False, showextrema=True)
        axes[1][i].set_title(f'Violin Plot of {title}')

        axes[1][i].scatter(1, medians[i], marker='o', color='white', s=40, zorder=3)
        axes[1][i].vlines(1, Q1[i], Q3[i], color='k', linestyle='-', lw=20)
        axes[1][i].vlines(1, lower_adjacent[i], upper_adjacent[i], color='k', linestyle='-', lw=3)
        axes[1][i].text(1, lower_adjacent[i], f'lower:{lower_adjacent[i]}', horizontalalignment='center')
        axes[1][i].text(1, upper_adjacent[i], f'upper:{upper_adjacent[i]}', horizontalalignment='center')

        axes[2][i].axis('off')
        axes[2][i].set_title(f'{lower_adjacent[i]} >) {title} (> {upper_adjacent[i]}')
        mask = np.logical_or(np_data[:, col] < lower_adjacent[i], np_data[:, col] > upper_adjacent[i])
        outliers_mask = np.logical_or(outliers_mask, mask)
        img = img_rgb.copy()
        for item in np_data[mask]:
            x1, y1 = int(item[0]), int(item[1])
            x2, y2 = int(item[0] + item[2]), int(item[1] + item[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
        axes[2][i].imshow(img)

    valid_data = np_data[np.logical_not(outliers_mask)]
    model = KMeans(n_clusters=6, max_iter=100)
    clusters = model.fit_predict(valid_data[:, 0].reshape(-1, 1))
    for i, m in enumerate(Counter(clusters).most_common()[:3]):
        cluster_data = valid_data[clusters == m[0]]
        xmin, xmax = cluster_data[:, 0].min(), cluster_data[:, 0].max()
        xmin, xmax = int(xmin - 2 * medians[0]), int(xmax + 2 * medians[0])

        img = np.zeros(img_rgb.shape, dtype=np.uint8)
        img[:] = [158, 164, 158]
        img[:, xmin:xmax, :] = img_rgb[:, xmin:xmax, :]

        for item in cluster_data:
            x1, y1 = int(item[0]), int(item[1])
            x2, y2 = int(item[0] + item[2]), int(item[1] + item[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
        axes[3][i].set_title(f'Count: {m[1]}')
        axes[3][i].imshow(img)
        axes[3][i].axis('off')


# test_samples = ['./ladder_39.png', './632306671_40.jpg']
# widgets.interact_manual(
#     extract_blackhole_rect,
#     imgpath=widgets.Dropdown(options=[(p.split('/')[-1][:-4], p) for p in test_samples]),
#     thresh=widgets.IntSlider(min=1, max=60, value=15),
#     kernel=widgets.IntSlider(min=1, max=16, value=3),
#     iterations=widgets.IntSlider(min=1, max=5, value=1),
#     iqr=widgets.FloatSlider(min=0.5, max=2.5, value=1.5)
# );

