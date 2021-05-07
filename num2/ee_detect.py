#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2



class Info:
    def __init__(self, width, height):
        self.ROI_spaceX = 20
        self.ROI_spaceY = 300
        self.width = width
        self.height = height

        self.leftUP = 100
        self.rightUP = 600

        self.src = np.float32(
            [(self.leftUP, self.ROI_spaceY), (self.rightUP, self.ROI_spaceY), (self.width, self.height), (0, self.height)])
        self.dst = np.float32([(0, 0), (self.width, 0),
                               (self.width, self.height), (0, self.height)])


class Line:
  

    def __init__(self, x1, y1, x2, y2):

        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)

        self.slope = self.get_slope()
        self.bias = self.get_bias()

    def get_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def get_bias(self):
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def set_coords(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, img, color=[255, 0, 0], thickness=10):
        cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)


def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    허프 라인 인식
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """
    차선이미지 덧그리기 (투명도 조절해서)
    """
    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))

    return cv2.addWeighted(initial_img, a, img, b, c)


def compute_lane_from_candidates(line_candidates, img_shape):
    """
    대표라인 찾기
    """

    # 기울기로 분류
    # pos_lines = [l for l in line_candidates if l.slope >
    #             0 and l.x1 < (img_shape[1] // 2)]
    # neg_lines = [l for l in line_candidates if l.slope <
    #             0 and l.x1 > (img_shape[1] // 2)]

    left_lines = [l for l in line_candidates if l.x1 < img_shape[1] // 2]
    right_lines = [l for l in line_candidates if l.x1 > img_shape[1] // 2]

  

    # left 라인들로 추정되는 애들의 절편값들의 평균
    left_bias = np.median([l.bias for l in left_lines]).astype(int)
    # 기울기들의 평균
    left_slope = np.median([l.slope for l in left_lines])

    # 화면에 꽉차게 좌표 설정
    if left_slope < 0:
        x1, y1 = 0, left_bias
        x2, y2 = -np.int32(np.round(left_bias / left_slope)), 0

    # 결국 위와 똑같지만, 선을 그릴때, 좌표로 점을 찍을거기때문에 굳이 이렇게 x, y 좌표 설정해줘야함
    else:
        x1, y1 = 0, left_bias
        x2, y2 = np.int32(np.round(480 / left_slope)) - \
            np.int32(np.round(left_bias / left_slope)), 480

    left_lane = Line(x1, y1, x2, y2)




    right_bias = np.median([l.bias for l in right_lines]).astype(int)
    right_slope = np.median([l.slope for l in right_lines])

    if right_slope < 0:
        x1, y1 = 0, right_bias
        x2, y2 = -np.int32(np.round(right_bias / right_slope)), 0

    else:
        x1, y1 = 0, right_bias
        x2, y2 = np.int32(np.round(480 / right_slope)) - \
            np.int32(np.round(right_bias / right_slope)), 480

    print(x1, y1, x2, y2)
    right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane


def get_lane_lines(img_bird, solid_lines=True):
    """
    차선 얻기
    """

    # gray scailing
    img_gray = cv2.cvtColor(img_bird, cv2.COLOR_BGR2GRAY)

    # blur 처리
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)

    # 캐니엣지
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)

    # 허프 라인 검출
    detected_lines = hough_lines_detection(img=img_edge,
                                           rho=2,
                                           theta=np.pi / 180,
                                           threshold=1,
                                           min_line_len=15,
                                           max_line_gap=5)

    # 허프변환을 통해 얻은 선들의 좌표를 Line함수에 입력해서 기울기와 절편 계산
    detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3])
                      for l in detected_lines]



    # 라인결정(solid_lines)를 선택한 경우, 라인들의 기울기와 절편을 통해 라인별 Line class들을 후보군에 포함시키기
    if solid_lines:
        candidate_lines = []
        for line in detected_lines:
            # 30도에서 60도 사이의 각만 넣음 (수치 조정 중. 원랜 0.5 ~ 2)
            if 0.2 <= np.abs(line.slope) <= 15:
                candidate_lines.append(line)

        # 왼쪽 대표라인, 오른쪽 대표라인의 좌표값이 있는 class를 받아옴
        lane_lines = compute_lane_from_candidates(
            candidate_lines, img_gray.shape)
    else:
        # solid_lines = False로 되어있으면, 그냥 허프변환한 결과만 그대로 도출
        lane_lines = detected_lines

    return lane_lines



def birdeye_view(img_main, info):
    Perspect = cv2.getPerspectiveTransform(info.src, info.dst)
    Perspect_back = cv2.getPerspectiveTransform(info.dst, info.src)
    img_bird = cv2.warpPerspective(
        img_main, Perspect, dsize=(info.width, info.height), flags=cv2.INTER_LINEAR)

    return img_bird, Perspect_back



# 불필요한 연산을 줄이기위해 (차선 이외 구역에있는 건물이라던지, 장애물도 선으로 인식하면 연산량이 많아지니깐)
# 관심영역설정을 따로 해줘야하는데 Birdeye View를 쓰기때문에 필요없음
def region_of_interest(img, vertices):

    # 마스크로 쓸 이미지
    mask = np.zeros_like(img)

    # 이미지 채널 고려
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # 사다리꼴 만들어주기
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image, mask


def img_processing(img_main, width, height):
    
    # 음.... ros.spin()이 콜백함수와 별개로 병렬로 작동하며, 이미지 토픽을 무대포로 계속받아와서 자꾸 프로그램이 다운됨... 그래서 해결해볼라고 
    # 일단 flag 추가해봤는데 안됨. 그래서 Cpp에 있는 ROS::spinOnce()함수 이용해서 해결해볼라거
    flag = 0

    # 클래스 인스턴스
    info = Info(width, height)

    # 버드아이뷰를 위한거
    img_bird, Perspect_back = birdeye_view(
        img_main, info)
   
    '''
    vertices = np.array(
        [[(0, height), (info.ROI_spaceX, info.ROI_spaceY), (width - info.ROI_spaceX, info.ROI_spaceY), (width, height)]], dtype=np.int32)
    img_roi, _ = region_of_interest(img_main, vertices)
    '''
    # 라인 추론
    lane_lines = []

    inferred_lanes = get_lane_lines(img_bird, True)
    lane_lines.append(inferred_lanes)

    lane_lines = lane_lines[0]

    #len(lane_lines)  # 이걸 이용해서 대표라인이 한쪽만 검출되는경우(왼쪽만 검출되는 경우) 예외설정같은거 해줄예정

    # 선 그리기
    img_line = np.zeros(shape=(height, width))

    for lane in lane_lines:
        lane.draw(img_bird)

    # 이미지 덧그리기
    img_blend = weighted_img(img_line, img_bird, a=0.8, b=1., c=0.)

    # 허프변환에서 되돌리기
    img_bird_back = cv2.warpPerspective(
        img_blend, Perspect_back, (width, height))



    flag = 1

    # 원래는 img_bird대신 img_blend를 리턴해줘야하는데 지금은 테스트를 위해 img_bird를 리턴! (그래서 메인문가보면 이 함수로 리턴받을 변수명으로 img_blend라 해놓음)
    return img_bird_back, img_bird
