#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import sys
import cv2, random, math, time
import cv2 as cv
import line_detect


Width = 640
Height = 360
Offset = 250

# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2

    cv2.rectangle(img, (lpos - 5, 15 + offset),
                       (lpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset),
                       (rpos + 5, 25 + offset),
                       (0, 255, 0), 2)   
    return img

# You are to find "left and light position" of road lanes
def process_image(frame,lpos,rpos):    
    #lpos, rpos = 100,500    

    global Offset
    
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)
    
    return (lpos, rpos), frame


def draw_steer(image, f_steer_angle, l_steer_angle, r_steer_angle):
    global Width, Height, arrow_pic


    arrow_pic = cv2.imread('steer_arrow.png', cv2.IMREAD_COLOR)

    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height/2
    arrow_Width = (arrow_Height * 462)/728
    

    #matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), -(180-((steer_angle) * 1.5)), 0.7)    
    matrix=angle_control(f_steer_angle, l_steer_angle, r_steer_angle, origin_Width, steer_wheel_center)
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = image[arrow_Height: Height, (Width/2 - arrow_Width/2) : (Width/2 + arrow_Width/2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow_pic)
    image[(Height - arrow_Height): Height, (Width/2 - arrow_Width/2): (Width/2 + arrow_Width/2)] = res

    #cv2.imshow('steer', image)

def angle_control(f_steer_angle, l_steer_angle, r_steer_angle, Width, center):
    kp = 0.2

    if r_steer_angle <=360:
        
        # Add
        r_angle= r_steer_angle * kp
        if r_angle > 20:
            r_angle = 20
        # /Add

        matrix = cv2.getRotationMatrix2D((Width/2, center), 180 - (r_angle), 0.7)
    
    else :
        if l_steer_angle >= 180:
            l_angle= l_steer_angle * kp
            l_angle = -20
            # /Add

            matrix=cv2.getRotationMatrix2D((Width/2,center), 180+(l_angle), 0.7)
        

        else :
            if 300 < f_steer_angle < 360:
                m_angle= f_steer_angle * kp

                # Add
                m_angle = 0
                # /Add

                matrix = cv2.getRotationMatrix2D((Width/2, center), m_angle+10, 0.7)
            else:
                f_angle=f_steer_angle
                matrix = cv2.getRotationMatrix2D((Width/2, center), 180-((f_angle) * 1.5), 0.7)


    return matrix
# You are to publish "steer_anlge" following load lanes
if __name__ == '__main__':
    cap = cv2.VideoCapture('kmu_track.mkv')
    time.sleep(3)

    while not rospy.is_shutdown():
        ret, image = cap.read()
        # pos, frame = process_image(image)
        
        # steer_angle = 0
        # draw_steer(frame, steer_angle)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

