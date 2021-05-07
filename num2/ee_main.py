#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
1. 이미지 받아오기
2. 영상처리
3. 차선인식
4. 회전계산
5. 모터에 퍼블리시
Matrix
ndarray
'''

import numpy as np
import cv2
import time

from ee_detect import img_processing

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage


FPS = 30
WAIT = 1000 // FPS


class AD():
    def __init__(self):
        self._sub = rospy.Subscriber(
            '/raspicam_node/image/compressed', CompressedImage, self.callback, queue_size=1, buff_size = 2 ** 24)
        self.bridge = CvBridge()
        self.flag = 1

    def callback(self, image_msg):

        np_arr = np.fromstring(image_msg.data, np.uint8)
        img_fliped = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_fliped = cv2.cvtColor(img_fliped, cv2.COLOR_RGB2BGR)
        height, width = img_fliped.shape[:2]

        img_main = cv2.flip(img_fliped, -1)

        img_blend, img_bird = img_processing(img_main, width, height)
        cv2.imshow("blend", img_blend)
        cv2.imshow("asdf", img_bird)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

    def main(self):
        while True:
            if self.flag == 0:
                time.sleep(100)
                continue

            rospy.spin()


if __name__ == '__main__':
    rospy.init_node('AD')
    node = AD()
    node.main()
