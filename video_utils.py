import cv2

def video_frame_generator(video_path,resize= False):
    cap = cv2.VideoCapture(video_path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if not cap.isOpened():
        print("Video reader initialization failed.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame,(int(w/2),int(h/2)))
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