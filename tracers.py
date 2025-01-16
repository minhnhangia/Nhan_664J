import cv2
import numpy as np


class CarTracer:
    def __init__(self, bg, video_height, video_width):
        self.bg = cv2.imread(bg)
        self.bg_height, self.bg_width, _ = self.bg.shape
        self.tracks = []
        self.video_height = video_height
        self.video_width = video_width
        self.bg = cv2.resize(self.bg, (self.video_width, self.video_height))    ### resize background img
        self.previous_track = None
        self.lower_white = np.array([0, 0, 200])  # HSV lower bound for white
        self.upper_white = np.array([180, 50, 255])  # HSV upper bound for white

    def plot_contours(self, frame):
        # detects white regions, filters them based on area, and draws bounding boxes
        contour_boxes = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)

        ### hierarchy to focus on parent contours, filtering out irrelevant contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(f"Hierarchy: {hierarchy}")
        for i, contour in enumerate(contours):
        # Use hierarchy to filter top-level contours
            if hierarchy[0][i][3] == -1:  # Check if the contour has no parent
                area = cv2.contourArea(contour)
                if 200 < area < 300:  # Filter based on area
                    cx, cy, cw, ch = cv2.boundingRect(contour)
                    contour_boxes.append((cx, cy, cw, ch))
                    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 255, 255), 2)
        return contour_boxes

    def trace(self, results_queue):
        """
        results[0]: Original frame (unmodified).
        results[1]: Annotated frame with bounding boxes and labels.
        results[2]: Normalized bounding box coordinates (xywhn).
        results[3]: Class IDs (cls).
        results[4]: Confidence scores (conf).
        """
        while True:
            results = results_queue.get()
            if results is None:
                break

            frame = results[0]
            bounding_boxes = self.plot_contours(frame)  # contours of white car
            for result, class_id, conf in zip(results[2].numpy(), results[3].numpy(), results[4].numpy()):
                if class_id != 3:  # ignore if not car
                    continue

                x_center, y_center, w_norm, h_norm = result
                # convert to pixel coordinates
                px = int(x_center * self.video_width)
                py = int(y_center * self.video_height)
                box_width = int(w_norm * self.video_width)
                box_height = int(h_norm * self.video_height)

                # Match with contours
                for cx, cy, cw, ch in bounding_boxes:
                    if cx <= px <= cx + cw and cy <= py <= cy + ch and conf > 0:
                        cv2.rectangle(frame, (px - box_width//2, py - box_height//2), (px + box_width//2, py + box_height//2), (0, 255, 0), 2)  # YOLO
                        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)  # Contours

                        if self.previous_track is not None:
                            cv2.line(self.bg, self.previous_track, (px, py), (0, 255, 0), 2)
                        self.previous_track = (px, py)

            # for result in results[2].numpy():
            #     x, y, _, _ = result
            #     px = int(x * self.video_width)
            #     py = int(y * self.video_height)
            #     if self.previous_track is not None:
            #         cv2.line(
            #             self.bg,
            #             self.previous_track,
            #             (px, py),
            #             (0, 255, 0),
            #             2,
            #         )
            #     self.previous_track = (px, py)

            cv2.imshow("White car trace", self.bg)
            cv2.imwrite("./submission/output.png", self.bg)
            cv2.imshow("Detections Frame", results[1])
            cv2.imshow("Filtered Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
