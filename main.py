import cv2
import numpy as np
import argparse
from json import load, dump
import torch
from datetime import time,datetime,timedelta
from pathlib import Path

from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
from OC_SORT.yolox.utils.visualize import plot_tracking
from OC_SORT.trackers.tracking_utils.timer import Timer
from gui_utils import select_lines, select_four_rois, select_best_feature_in_roi
from video_utils import VideoWriter, video_frame_generator
from draw_utils import plot_lines, plot_points,draw_label
from math_utils import ViewTransformer
from car_state import CarTrack
from collections import defaultdict


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
        "--select_road_lines_by_gui", action="store_true", help="Select road lines by selecting points on frame, if false, points are read from config.json",
    )
    parser.add_argument(
        "--update_lines_in_config", action="store_true", help="Update the road line coords in config.json with new ones selected via mouse", 
    )
    parser.add_argument(
        "--select_ref_points_by_gui", action="store_true", help="Select reference points by selecting points on frame, if false, points are read from config.json",
    )
    parser.add_argument(
        "--update_ref_points_in_config", action="store_true", help="Update the reference points coords in config.json with new ones selected via mouse", 
    )
    parser.add_argument(
        "--resize_video_by_half", action="store_true", help="Update the reference points coords in config.json with new ones selected via mouse")

    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_arguments()

    now = datetime.now() # generated to use timedelta addition
    time_begin = time(13,50)
    datetime_begin = datetime.combine(now,time_begin)
    time_passed_in_seconds = 0
    video_fps = 30
    result_list = []
    
    frame_generator = video_frame_generator(args.source_video_path,args.resize_video_by_half)

    ocsort_tracker = OCSort(max_age=30,
                            det_thresh=0.6,
                            iou_threshold=0.3,
                            use_byte=True)
    
    min_box_area = 20 # discard possible false positive or too far detections which cannot be tracked optimally

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
    config_path = Path(args.source_video_path).stem+"_config"
    config_path = Path(config_path).with_suffix(".json")
    
    if not config_path.exists():
        with open(config_path,"w") as fp:
            dump({},fp)
    
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
            with open(config_path,"r+") as fp:
                config = load(fp)
                config["ref_points_for_video_stabilization"] = p0
                fp.seek(0)
                config = dump(config, fp,indent=1)
                fp.truncate()
    else:
        with open(config_path,"r") as fp:
            config = load(fp)
            p0 = config["ref_points_for_video_stabilization"]
 
    frame = plot_points(frame,p0)
    p0 = np.array(p0, dtype=np.float32)
    p0 = np.expand_dims(p0,axis=1)

    track_ids_to_tracks_dict = {}

    # Select road vertical line coordinates from gui or config
    if args.select_road_lines_by_gui:
        road_vertical_line_coords = select_lines(frame,"Select road vertical reference lines from leftmost to rightmost. Press 'q' to quit.")
        if args.update_lines_in_config:
            with open(config_path,"r+") as fp:
                config = load(fp,)
                config["road_vertical_line_coords"] = road_vertical_line_coords
                fp.seek(0)
                config = dump(config, fp, indent=1)
                fp.truncate()
    else:
        with open(config_path,"r") as fp:
            config = load(fp)
            road_vertical_line_coords = config["road_vertical_line_coords"]

    height,width,_ = frame.shape
    top_horizontal_line_y = int(0.7*height)
    bottom_horizontal_line_y = int(0.8*height)
    top_horizontal_line_coords = [[0,top_horizontal_line_y],[width,top_horizontal_line_y]]
    bottom_horizontal_line_coords = [[0,bottom_horizontal_line_y],[width,bottom_horizontal_line_y]]

    # Load pretrained object detection model
            
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s6',
                        device="0" if torch.cuda.is_available() else "cpu")
            
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s6',
    #                     device="cpu")
            
    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s6.onnx')
    
    timer = Timer() # This timer is used for FPS calculation purpose, not car state timestamp
    timer.tic()
    # generate a grayscale copy of first frame for using in Lucas Canade optical flow
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0_0 = p0.copy() # store initial coordinates of selected reference points to later calculate location transformation

    states_to_trackids = defaultdict(list)

    # Detection and tracking inference loop
    with VideoWriter(args.target_video_path, fps = 30) as video_writer:
        for frame_id, frame in enumerate(frame_generator):
            time_passed_in_seconds = (frame_id+1)/video_fps
            current_timestamp = datetime_begin + timedelta(seconds=time_passed_in_seconds)

            # Calculate optical flow for selected reference points and apply spatial transformation to road lane coordinates
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            view_transformer = ViewTransformer(source=p0_0, target=p1)

            # update line coordinates with new estimated coordinates
            updated_coordinates = []
            for coord in road_vertical_line_coords:
                updated_coord = view_transformer.transform_points(np.array(coord))
                updated_coord = [[int(j) for j in i] for i in updated_coord]
                updated_coordinates.append(updated_coord)

            good_new = p1[st == 1] 
            old_gray = frame_gray.copy()    # update old_gray and p0 for next iteration
            p0 = good_new.reshape(-1, 1, 2) 
            
            object_predictions = model(frame,size=1280)
            outputs = object_predictions.pred[0]
            if outputs is not None:   # run tracker on bbox predictions
                online_targets = ocsort_tracker.update(outputs, [frame.shape[0], frame.shape[1]], (frame.shape[0], frame.shape[1]))
                online_tlwhs = []     # output of the tracker is a np.array as [[x1,y1,x2,y2,track_id],[x1,y1,x2,y2,track_id],...]
                online_ids = []
                for t in online_targets:
                    tid = t[4]
                    car_bottom_coord = (int((t[0]+t[2])/2),int(t[3]))
                    
                    if track_ids_to_tracks_dict.get(tid): # add coordinate to the CarTrack objects history
                        track_ids_to_tracks_dict[tid].add_coordinate(t[:4],updated_coordinates)
                    else: # Initialize a CarTrack class instance which keeps the coordinate history
                        track_ids_to_tracks_dict[tid] = CarTrack(id,t[:4])
                    vert_direction = track_ids_to_tracks_dict[tid].check_direction(top_horizontal_line_y,bottom_horizontal_line_y)  # calculate the current vertical movement state using history
                    lane_change_state = track_ids_to_tracks_dict[tid].check_lane_change()  # calculate the current lane change state using history
                    frame = draw_label(frame,t,vert_direction)
                    frame = draw_label(frame,t,lane_change_state)
                    
                    
                    if len(vert_direction) > 0 and (tid not in states_to_trackids[vert_direction]): # could be also stationary car which is returned from
                                                # function as "" empty string for generic draw_label trick
                        car_data = {
                            'car_id': tid,
                            'timestamp': str(current_timestamp),
                            'state': vert_direction.lower()
                        }
                        result_list.append(car_data)
                        states_to_trackids[vert_direction].append(tid)
                    if len(lane_change_state) > 0 and (tid not in states_to_trackids[lane_change_state]):
                        car_data = {
                            'car_id': tid,
                            'timestamp': str(current_timestamp),
                            'state': lane_change_state.lower()
                        }
                        result_list.append(car_data)
                        states_to_trackids[lane_change_state].append(tid)

                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    if tlwh[2] * tlwh[3] > min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                
                timer.toc()
                fps = 1. / timer.average_time

                frame = plot_tracking(image=frame,
                              tlwhs=online_tlwhs,
                              obj_ids=online_ids,
                              frame_id=frame_id,
                              fps=fps)
            
            frame = plot_lines(frame, updated_coordinates)
            frame = plot_lines(frame, [top_horizontal_line_coords,bottom_horizontal_line_coords])
            frame = plot_points(frame,p0.squeeze().tolist())
            cv2.imshow("Car Direction Counting", frame)
            video_writer.write_frame(frame)
            delay = 1
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
            timer.tic()
        cv2.destroyAllWindows()
    
    result_json_path = Path(args.source_video_path).stem+"_result"
    result_json_path = Path(result_json_path).with_suffix(".json")
    with open(result_json_path, "w") as fs:
        dump(result_list,fs)
