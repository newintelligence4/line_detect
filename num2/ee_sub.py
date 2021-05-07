#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np

img_path = './01.png'
ori_img = cv2.imread(img_path)

src = []

# mouse callback handler


def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        img = ori_img.copy()

        src.append([x, y])

        for xx, yy in src:
            cv2.circle(img, center=(xx, yy), radius=5, color=(
                0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

        cv2.imshow('img', img)

        # perspective transform
        if len(src) == 4:
            src_np = np.array(src, dtype=np.float32)

            print('src = ', src_np)

            width = 640
            height = 480 - 250

            dst_np = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ], dtype=np.float32)

            print('dst_np = ', dst_np)

            M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
            result = cv2.warpPerspective(ori_img, M=M, dsize=(width, height))

            cv2.imshow('result', result)


# main
cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_handler)

cv2.imshow('img', ori_img)
cv2.waitKey(0)
