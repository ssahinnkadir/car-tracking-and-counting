import cv2
import argparse
from json import load, dump
import torch
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
from OC_SORT.yolox.utils.visualize import plot_tracking
from OC_SORT.trackers.tracking_utils.timer import Timer

import numpy as np

def video_frame_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    w = int(w/2.5)
    h = int(h/2.5)
    
    if not cap.isOpened():
        print("Video reader initialization failed.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,(w,h))
        yield frame

    cap.release()


class VideoWriter:
    def __init__(self, output_path, fps):
        self.output_path = output_path
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def write_frame(self, frame):
        if self.out is None:
            height, width, _ = frame.shape
            self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (width, height))
        self.out.write(frame)

    def release(self):
        if self.out is not None:
            self.out.release()

class ViewTransformer:
	def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
		source = source.astype(np.float32)
		target = target.astype(np.float32)
		self.m = cv2.getPerspectiveTransform(source, target)

	def transform_points(self, points: np.ndarray) -> np.ndarray:
		if points.size == 0:
			return points

		reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
		transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
		return transformed_points.reshape(-1, 2)


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

def plot_lines(img,line_coord_list):
    for coord in line_coord_list:
        cv2.line(img, *coord,  color=(255, 0, 0), thickness=2)
    return img
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Car Direction Counting via Tracking"
    )
    parser.add_argument(
        "source_video_path",
        nargs="*",
        help="Path to the source video file",
        default="./data/video1.MOV",
        type=str,
    )
    parser.add_argument(
        "target_video_path",
        nargs="*",
        help="Path to the target video file (output)",
        default="./data/video1_out.MOV",
        type=str,
    )
    parser.add_argument(
        "--select_road_lines_by_gui", default=False, help="Select road lines by selecting points on frame, if false, points are read from config.json", type=bool
    )
    parser.add_argument(
        "--update_lines_in_config", default=False, help="Update the road line coords in config.json with new ones selected via mouse", type=bool
    )
    parser.add_argument(
        "--select_ref_points_by_gui", default=False, help="Select road lines by selecting points on frame, if false, points are read from config.json", type=bool
    )
    parser.add_argument(
        "--update_ref_points_in_config", default=False, help="Update the road line coords in config.json with new ones selected via mouse", type=bool
    )

    return parser.parse_args()

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
    
def plot_points(frame,coords):
    for coord in coords:
        x,y = coord
        x=int(x)
        y=int(y)
        frame = cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    return frame

if __name__ == "__main__":
    args = parse_arguments()
    
    frame_generator = video_frame_generator(args.source_video_path)

    # ocsort_tracker = OCSort(max_age=30,
    #                         det_thresh=0.6,
    #                         iou_threshold=0.3,
    #                         use_byte=True)
    
    min_box_area = 20   

    frame = next(frame_generator)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Parameters for cv2.goodFeaturesToTrack 
    feature_params = dict(maxCorners = 27,
                       qualityLevel = 0.01,
                       minDistance = 10,
                       blockSize = 7)
    
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                              10, 0.03))
    
    if args.select_ref_points_by_gui:
        selected_rois = select_four_rois(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = []
        for roi in selected_rois:
            coord  = select_best_feature_in_roi(gray_frame,roi,feature_params=feature_params)
            if coord == None:
                x1, y1, x2, y2 = roi
                coord = (int((x1+x2)/2),int((y1+y2)/2))
            p0.append(coord)
        if args.update_ref_points_in_config:
            with open("config.json","r+") as fp:
                config = load(fp,)
                config["ref_points_for_video_stabilization"] = p0
                fp.seek(0)
                config = dump(config, fp,indent=1)
                fp.truncate()
    else:
        with open("config.json","r") as fp:
            config = load(fp)
            p0 = config["ref_points_for_video_stabilization"]
 

    frame = plot_points(frame,p0)
    p0 = np.array(p0, dtype=np.float32)
    p0 = np.expand_dims(p0,axis=1)

    # Select road vertical line coordinates from gui or config
    if args.select_road_lines_by_gui:
        road_vertical_line_coords = select_lines(frame,"Select road vertical reference lines from leftmost to rightmost. Press 'q' to quit.")
        if args.update_lines_in_config:
            with open("config.json","r+") as fp:
                config = load(fp,)
                config["road_vertical_line_coords"] = road_vertical_line_coords
                fp.seek(0)
                config = dump(config, fp, indent=1)
                fp.truncate()
    else:
        with open("config.json","r") as fp:
            config = load(fp)

            road_vertical_line_coords = config["road_vertical_line_coords"]

    # # Load pretrained object detection model
    # model = torch.hub.load('ultralytics/yolov5',
    #                     'yolov5s6',
    #                     device="0" if torch.cuda.is_available() else "cpu")
    

    timer = Timer()
    
    timer.tic()

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # p0 = cv2.goodFeaturesToTrack(old_gray, 27, 0.01, 10)

    color = np.random.randint(0, 255, (100, 3))

    p0_0 = p0.copy()

    with VideoWriter(args.target_video_path, fps = 30) as video_writer:
        for frame_id, frame in enumerate(frame_generator):
            # frame = plot_points(frame,p0.squeeze().tolist())
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # p0 = np.array(p0, dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            view_transformer = ViewTransformer(source=p0_0, target=p1)

            updated_coordinates = []
            for coord in road_vertical_line_coords:
                updated_coord = view_transformer.transform_points(np.array(coord))
                updated_coord = [[int(j) for j in i] for i in updated_coord]
                updated_coordinates.append(updated_coord)

            good_new = p1[st == 1] 
            good_old = p0[st == 1]
            for i, (new, old) in enumerate(zip(good_new,  
                                       good_old)): 
                a, b = new.ravel()
                c, d = old.ravel()
                a,b,c,d = int(a) , int(b), int(c), int(d)
                frame = cv2.line(frame, (a, b), (c, d), 
                                color[i].tolist(), 2) 
                
                frame = cv2.circle(frame, (a, b), 5, 
                                color[i].tolist(), -1)
            
            
            
            # frame = plot_points(frame,good_new.squeeze().tolist())
            old_gray = frame_gray.copy() 
            p0 = good_new.reshape(-1, 1, 2) 
            
            # object_predictions = model(frame,size=1280)
            # outputs = object_predictions.pred[0]
            # if outputs is not None:
            #     online_targets = ocsort_tracker.update(outputs, [frame.shape[0], frame.shape[1]], (frame.shape[0], frame.shape[1]))
            #     online_tlwhs = []
            #     online_ids = []
            #     for t in online_targets:
            #         tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            #         tid = t[4]
            #         if tlwh[2] * tlwh[3] > min_box_area:
            #             online_tlwhs.append(tlwh)
            #             online_ids.append(tid)
                
            #     timer.toc()
            #     fps = 1. / timer.average_time

            #     frame = plot_tracking(image=frame,
            #                   tlwhs=online_tlwhs,
            #                   obj_ids=online_ids,
            #                   frame_id=frame_id,
            #                   fps=fps)
            
            
            
            frame = plot_lines(frame, updated_coordinates)
            cv2.imshow("Car Direction Counting", frame)
            video_writer.write_frame(frame)
            delay = 1
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
            timer.tic()
        cv2.destroyAllWindows()
