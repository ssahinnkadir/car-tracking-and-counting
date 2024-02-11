import cv2
import argparse

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

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    frame_generator = video_frame_generator(args.source_video_path)
    
    with VideoWriter(args.target_video_path, fps = 30) as video_writer:
        for frame in frame_generator:
            cv2.imshow("Car Direction Counting", frame)
            video_writer.write_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
