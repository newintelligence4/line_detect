import numpy as np
import cv2, random, math, time
import line_detect_roi

width = 640
height = 480
offset = 330
gap = 40

# draw lines
def draw_lines(img, lines):
    global offset

    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1+offset), (x2, y2+offset), color, 2)

    return img

#draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2

    cv2.rectangle(img, (lpos - 5, 15 + offset), (lpos + 5, 25 + offset), (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset), (rpos + 5, 25 + offset), (0, 255, 0), 2)
    cv2.rectangle(img, (center - 5, 15 + offset), (center + 5, 25 + offset), (0, 255, 0), 2)
    cv2.rectangle(img, (315, 15 + offset), (325, 25 + offset), (0, 0, 255), 2)

    return img

# left lines, right lines
def divide_left_right(lines):
    global width

    low_slope_threshold = 0
    high_slope_threshold = 10

    # calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2 - y1) / float(x2 - x1)

        if abs(slope) > low_slope_threshold and\
           abs(slope) < high_slope_threshold:

            slopes.append(slope)
            new_lines.append(line[0])
    # divide lines left to right
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line
        if (slope < 0) and (x2 < width/2 - 90):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > width/2 + 90):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines

# get average m, b of lines
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = float(x_sum) / float(size * 2)
    y_avg = float(y_sum) / float(size * 2)

    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b

# get lpos, rpos
def get_line_pos(lines, left=False, right=False):
    global width, height
    global offset, gap

    m, b = get_line_params(lines)

    x1, x2 = 0, 0
    if m == 0 and b == 0:
        if left:
            pos = 0
        if right:
            pos = width
    else:
        y = gap / 2
        pos = (y - b) / m

        b += offset
        x1 = (height - b) / float(m)
        x2 = ((height / 2) - b) / float(m)

    return x1, x2, int(pos)


def process_image(frame):
    global width
    global offset, gap
    # convert BGR to GrayScale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian blur filltering
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # canny edge
    edge = cv2.Canny(blur, 60, 70)
    # setting roi
    roi = edge[offset:offset+gap, 0:width]
    # Hough Transform
    all_lines = cv2.HoughLinesP(roi, 1, math.pi/180, 30, 30, 10)
    # divide left, right lines
    if all_lines is None:
        return (0, 640), frame

    left_lines, right_lines = divide_left_right(all_lines)
    # get center of lines
    lx1, lx2, lpos = get_line_pos(left_lines, left=True)
    rx1, rx2, rpos = get_line_pos(right_lines, right=True)

    frame = cv2.line(frame, (int(lx1), height), (int(lx2), height//2), (255, 0, 0), 3)
    frame = cv2.line(frame, (int(rx1), height), (int(rx2), height//2), (255, 0, 0), 3)
    # draw lines
    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)

    return (lpos, rpos), frame

def start():
    global image, width, height

    cap = cv2.VideoCapture('kmu_track.mkv')

    while True:
        ret, img = cap.read()
        pos, frame = process_image(img)
        
        center = (pos[0] + pos[1]) / 2
        angle = 320 - center
        steer_angle = angle * 0.4
        cv2. imshow('result',frame)
        # draw_steer(frame, steer_angle)
        
        if not ret:
            break
        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start()