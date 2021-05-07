import cv2
import sys
import math
import cv2 as cv
import numpy as np

cap=cv2.VideoCapture("kmu_track.mkv")


def roi(image):
    x=int(image.shape[1])
    y=int(image.shape[0])

    _shape=np.array([[int(0),int(0.85*y)],
                    [int(0.35*x),int(0.54*y)],
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
    #cv2.imshow('mid',masked_image)
    return masked_image

while(True):
    ret,src=cap.read()
    src=cv2.resize(src,(640,360))
    
    dst=cv.Canny(src,50,200,None,3)
    dst=roi(dst)
    

    cdst=cv.cvtColor(dst,cv.COLOR_GRAY2BGR)
    cdstP=np.copy(cdst)

    # lines=cv.HoughLines(dst,1,np.pi/180,150,None,0,0)

    # if lines is not None:
    #     for i in range(0,len(lines)):
    #         rho=lines[i][0][0]
    #         theta=lines[i][0][1]
    #         a=math.cos(theta)
    #         b=math.sin(theta)
    #         x0=a*rho
    #         y0=b*rho
    #         pt1=(int(x0+1000*(-b)),int(y0+1000*(a)))
    #         pt2=(int(x0-1000*(-b)),int(y0-1000*(a)))
    #         cv.line(cdst,pt1,pt2,(0,255,0),3,cv.LINE_AA)

    linesP=cv.HoughLinesP(dst,1,np.pi/180,30,None,50,10)

    if linesP is not None:
        for i in range(0,len(linesP)):
            l=linesP[i][0]
            cv.line(cdstP,(l[0],l[1]),(l[2],l[3]), (0, 0, 255),3,cv.LINE_AA)


    img_B, img_G, img_R = cv2.split(cdstP)

    y = 250

    nonzero_x = img_R[y, :].nonzero()[0]

    img_R = cv2.merge((img_R, img_R, img_R))

    if(abs(nonzero_x[0] - nonzero_x[-1]) < 70):
        cv2.imshow("asdf", img_R)
        continue

    x0 = nonzero_x[0]
    x1 = nonzero_x[-1]

    X=int(img_R.shape[1])
    Y=int(img_R.shape[0])

    cv2.line(img_R, (x0, y), (x0, y), (0, 255, 0), 20)
    cv2.line(img_R, (x1, y), (x1, y), (0, 255, 0), 20)

    midLane = (x1 + x0) / 2

    cv2.line(img_R, (midLane, y), (midLane, y), (0, 0, 255), 10) 
    cv2.line(img_R, (X/2,0),(X/2,Y),(255,0,0),5)

    print( x0, midLane, x1 )

    cv2.imshow("asdf", img_R)

    cv.imshow("source",src)
    # cv.imshow("Detected Lines(in red)-Standard Hough Line Transform",cdst)
    cv.imshow("Detected Lines(in red)-Probabilistic Line Transform",cdstP)

    if cv2.waitKey(0)& 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
