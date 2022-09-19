import cv2
import numpy as np

def get_lanes(source):
    first_frame = None
    roi = None
    video = cv2.VideoCapture(source)
    showCrossHair = None
    fromCenter = False
    lane_lines = None

    # read only first frame and extract lanes
    if first_frame is None:
        _, first_frame = video.read()

        # use keyboard and get ROI area
        roi = cv2.selectROI("Select area", first_frame, fromCenter, showCrossHair)

        # Crop ROI
        imCrop = first_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]




        # Perform dilation to increase size of white color
        kernel = np.ones((3, 3), np.uint8)
        first_frame = cv2.dilate(imCrop, kernel,iterations=1)

        #convert to gray scale
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Threshold the gray image to get only white colors
        white_lane_mask = cv2.inRange(gray, 215, 255)

        # bit wise and generate mask highway(without for loop)
        new_masked = cv2.bitwise_and(gray, gray, mask=white_lane_mask)

        #Apply threshold
        thresh, gray = cv2.threshold(new_masked, 158, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(gray, 0.3*thresh, thresh)
        lane_lines = cv2.HoughLinesP(edges, 2, np.pi/180, 30, minLineLength=15, maxLineGap=48)

    return lane_lines, roi

if __name__ == '__main__':
    video_path = "1.mp4"

    video = cv2.VideoCapture(video_path)
    lines, r = get_lanes(video_path)

    while True:
        check, frame = video.read()
        if check == False:
            break

        # draw lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1 + r[0], y1 + r[1]), (x2 +r[0], y2 +r[1]), (0, 255, 255), 3)
        # Display output
        cv2.imshow("Demo", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    pass
