import cv2


class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    def get_video_properties(self):
        # fetches video frame dimensions, need to resize??????????
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        return frame_height, frame_width

    def stream(self, frame_queue):
        # reads frame and pushes them to frame_queue
        if not self.cap.isOpened():
            raise Exception("Error opening video stream or file")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_queue.put(frame)

    def __del__(self):
        self.cap.release()
