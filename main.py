import cv2
import argparse
from json import load, dump

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

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    frame_generator = video_frame_generator(args.source_video_path)

    frame = next(frame_generator)
    


    if args.select_road_lines_by_gui:
        road_vertical_line_coords = select_lines(frame,"Select road vertical reference lines from leftmost to rightmost. Press 'q' to quit.")
        if args.update_lines_in_config:
            with open("config.json","r+") as fp:
                config = load(fp,)
                config["road_vertical_line_coords"] = road_vertical_line_coords
                fp.seek(0)
                config = dump(config, fp)
                fp.truncate()
    else:
        with open("config.json","r") as fp:
            config = load(fp)
            road_vertical_line_coords = config["road_vertical_line_coords"]
    
    with VideoWriter(args.target_video_path, fps = 30) as video_writer:
        for frame in frame_generator:
            frame = plot_lines(frame, road_vertical_line_coords)
            cv2.imshow("Car Direction Counting", frame)
            video_writer.write_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
