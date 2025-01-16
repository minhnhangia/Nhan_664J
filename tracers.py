import cv2
import numpy as np


class CarTracer:
    def __init__(self, bg, video_height, video_width):
        self.bg = cv2.imread(bg)
        self.bg_height, self.bg_width, _ = self.bg.shape
        self.tracks = []
        self.video_height = video_height
        self.video_width = video_width
        self.previous_track = None
        self.lower_white = np.array([0, 0, 200])  # HSV lower bound for white
        self.upper_white = np.array([180, 50, 255])  # HSV upper bound for white

    def plot_contours(self, frame):
        # detects white regions, filters them based on area, and draws bounding boxes
        contour_boxes = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                # draws bounding boxes around detected white regions
                cx, cy, cw, ch = cv2.boundingRect(contour)
                contour_boxes.append((cx, cy, cw, ch))
                cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
        return contour_boxes

    def trace(self, results_queue):
        while True:
            results = results_queue.get()
            frame = results[0]
            if results is None:
                break
            _ = self.plot_contours(frame)
            for result in results[2].numpy():
                x, y, _, _ = result
                px = int(x * self.video_width)
                py = int(y * self.video_height)
                if self.previous_track is not None:
                    cv2.line(
                        self.bg,
                        self.previous_track,
                        (px, py),
                        (0, 255, 0),
                        2,
                    )
                self.previous_track = (px, py)

            cv2.imshow("White car trace", self.bg)
            cv2.imwrite("./submission/output.png", self.bg)
            cv2.imshow("Detections Frame", results[1])
            cv2.imshow("Filtered Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
