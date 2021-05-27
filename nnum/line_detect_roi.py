import rospy
import cv2
import sys
import math
import time
import cv2 as cv
import numpy as np
import line_drive_if

def roi(image):
    x=int(image.shape[1])
    y=int(image.shape[0])

    
    _shape=np.array([[int(0),int(0.85*y)],
                   [int(0.30*x),int(0.54*y)],
                   [int(0.62*x),int(0.52*y)],
                   [int(x),int(0.7*y)],
                   [int(x),int(y)]])                
    mask=np.zeros_like(image)

    if len(image.shape)>2:
        channel_count=image.shape[2]
        ignore_mask_color=(255,)*channel_count
    else:
        ignore_mask_color=255 
        


    cv2.fillPoly(mask,np.int32([_shape]),ignore_mask_color)
    masked_image=cv2.bitwise_and(image,mask)
    cv2.imshow('mid',masked_image)
    return masked_image

def warpping(image):
    (h,w)=(image.shape[0],image.shape[1])

    source=np.float32([[100, h-160],[w-60, h-160],[w,h-130],[0,h-120]])
    
    des=np.float32([[0,0],[w,0],[w,h],[0,h]])

    transform_matrix=cv2.getPerspectiveTransform(source,des)
    minv=cv2.getPerspectiveTransform(des,source)
    _image=cv2.warpPerspective(image,transform_matrix,(w,h))
    
    return _image, minv

    
def sliding_window(image):
    nwindows = 9
    margin = 50
    minpix = 15

    #blur = cv2.GaussianBlur(image,(5, 5), 0)
    #_, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))
    #_, lane1 = cv2.threshold(, 145, 255, cv2.THRESH_BINARY)

    k =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    lane = cv2.dilate(image, k)

    cv2.imshow('ddddd',lane)
    histogram = np.sum(lane[lane.shape[0]//2:,:], axis=0)
    midpoint=np.int(histogram.shape[0]/2)
    leftx_current = np.argmax(histogram[ : midpoint])
    rightx_current = np.argmax(histogram[midpoint: ])+midpoint

    #window_height = np.int(lane.shape[0]/nwindows)
    window_height = 50
    nz= lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []

    lx, ly, rx, ry = [], [], [], []

    out_img = np.dstack((lane, lane, lane))*255

    for window in range(nwindows):
        win_yl = lane.shape[0] - (window+1)*window_height
        win_yh = lane.shape[0] - window*window_height


        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin

        cv2.rectangle(out_img, (win_xll, win_yl),(win_xlh, win_yh), (0, 255, 0),2)
        cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh),(0, 255, 0),2)

        good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nz[1][good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nz[1][good_right_inds]))

        
        lx.append(leftx_current)
        ly.append((win_yl+win_yh)/2)

        rx.append(rightx_current)
        ry.append((win_yl+win_yh)/2)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    lfit = np.polyfit(np.array(ly), np.array(lx),2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)

    out_img[nz[0][left_lane_inds],nz[1][left_lane_inds]]=[255,0,0]
    out_img[nz[0][right_lane_inds],nz[1][right_lane_inds]]=[0,0,255]

    cv2.imshow('viewer', out_img)






def sliding_window_y(image):
    nwindows = 20
    margin = 12
    minpix = 5

    #blur = cv2.GaussianBlur(image,(5, 5), 0)
    #_, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))
    #_, lane1 = cv2.threshold(, 145, 255, cv2.THRESH_BINARY)

    k =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    lane = cv2.dilate(image, k)

    cv2.imshow('ddddd',lane)
    histogram_y = np.sum(lane[:, lane.shape[1]//2:], axis=1)
    midpoint_y=np.int(histogram_y.shape[0]/2)
    highy_current = np.argmax(histogram_y[ : midpoint_y])
    lowy_current = np.argmax(histogram_y[midpoint_y: ])+midpoint_y

    #window_height = np.int(lane.shape[0]/nwindows)
    window_height = 50
    nz= lane.nonzero()

    high_lane_inds = []
    low_line_inds = []

    lx, ly, rx, ry = [], [], [], []

    out_img = np.dstack((lane, lane, lane))*255

    for window in range(nwindows):
        win_xl = lane.shape[1] - (window+1)*window_height
        win_xh = lane.shape[1] - window*window_height


        win_xll = highy_current - margin
        win_xlh = highy_current + margin
        win_xrl = lowy_current - margin
        win_xrh = lowy_current + margin

        cv2.rectangle(out_img, (win_xll, win_xl),(win_xlh, win_xh), (200, 125, 0),2)
        cv2.rectangle(out_img, (win_xrl, win_xl), (win_xrh, win_xh),(200, 125, 0),2)

        good_high_inds = ((nz[1] >= win_xl)&(nz[1] < win_xh)&(nz[0] >= win_xll)&(nz[0] < win_xlh)).nonzero()[0]
        good_low_inds = ((nz[1] >= win_xl)&(nz[1] < win_xh)&(nz[0] >= win_xrl)&(nz[0] < win_xrh)).nonzero()[0]

        high_lane_inds.append(good_high_inds)
        low_line_inds.append(good_low_inds)

        if len(good_high_inds) > minpix:
            highy_current = np.int(np.mean(nz[1][good_high_inds]))
        if len(good_low_inds) > minpix:
            lowy_current = np.int(np.mean(nz[1][good_low_inds]))

        
        lx.append(highy_current)
        ly.append((win_xl+win_xh)/2)

        rx.append(lowy_current)
        ry.append((win_xl+win_xh)/2)

    high_lane_inds = np.concatenate(high_lane_inds)
    low_line_inds = np.concatenate(low_line_inds)

    lfit = np.polyfit(np.array(ly), np.array(lx),2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)

    out_img[nz[0][high_lane_inds],nz[1][high_lane_inds]]=[150,0,150]
    out_img[nz[0][low_line_inds],nz[1][low_line_inds]]=[0,150,150]

    cv2.imshow('viewer1', out_img)


if __name__=='__main__':

    cap=cv2.VideoCapture("kmu_track.mkv")
    

    while (True):
        ret,src=cap.read()
        src=cv2.resize(src,(640,480))
       
        dst=cv.Canny(src,50,200,None,3) # getting outlines
        # dst=roi(src)
        cdst, minv = warpping(dst)
        
        #cdstP=cv.cvtColor(cdst,cv.COLOR_GRAY2BGR)
       
        cv2.imshow('image!',cdst)
        window= sliding_window(cdst)
        #windowy = sliding_window_y(cdst)
 

        
        '''
        pos, frame = line_drive_if.process_image(img_R, x0, x1)

        f_steer_angle = midLane
        l_steer_angle = x0
        r_steer_angle = x1
       
        line_drive_if.draw_steer(frame, f_steer_angle, l_steer_angle, r_steer_angle )

       

        #print( x0, midLane, x1 )

        

        cv.imshow("source",src)
        # cv.imshow("Detected Lines(in red)-Standard Hough Line Transform",cdst)
        cv.imshow("Detected Lines(in red)-Probabilistic Line Transform",cdstP)
        cv2.imshow("asdf", img_R)
        '''
        if cv2.waitKey(33)& 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
