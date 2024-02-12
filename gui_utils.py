import cv2
from draw_utils import plot_lines

def select_line(frame,window_msg):
    points = []
    quit_pressed = False
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # add clicked point coordinates to the points list as tuple
            points.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(window_msg, frame)

    # create window and connect to the callback function
    cv2.namedWindow(window_msg)
    cv2.setMouseCallback(window_msg, mouse_callback)

    while True:
        cv2.imshow(window_msg, frame)
        key = cv2.waitKey(1) & 0xFF
        # exit when 'q' is pressed or when enough point to create a line is selected
        if key == ord("q") or len(points) == 2:
            break
    if len(points) != 2: # quitted by 'q' press
        cv2.destroyAllWindows()
        quit_pressed = True
        return None, quit_pressed
    print("Selected line end coordinates:", points)

    frame = plot_lines(frame, [points,])
    return points, quit_pressed

def select_lines(img,window_msg):
    lines = []
    quit_pressed = False

    while(not quit_pressed):
        line, quit_pressed = select_line(img,window_msg)
        if line is not None:
            lines.append(line)
    return lines


def select_four_rois(image):
    """ This function is for selecting 4 reference point to track for 
        performing video stabilization (adaptive road line shifting)
    """
    rois = []

    for _ in range(4):
        roi = cv2.selectROI("Select ROI", image)
        (x, y, w, h) = roi
        x1y1x2y2  = (x, y, x + w, y + h)
        rois.append(x1y1x2y2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Select ROIs and press ENTER", image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    return rois

def select_best_feature_in_roi(gray_frame, roi, feature_params):
    """ It is recommended to use goodFeaturesToTrack method of cv2 to
        select candidate reference points for tracking. That's why we first
        select a roi (which we are sure there is no dynamic object inside
        such as leaf or car) and then utilize goodFeaturesToTrack to determine
        best feature point inside that roi
    """

    x1, y1, x2, y2 = roi
    roi_gray = gray_frame[y1:y2, x1:x2]
    corners = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

    if corners is not None:
        # select best reference point from candidates
        best_corner = corners[0][0]
        return (int(best_corner[0]) + x1, int(best_corner[1]) + y1)
    else:
        return None
    
