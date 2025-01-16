from ultralytics import YOLO
import cv2


class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame_queue, results_queue):
        # retrieves frames from frame_queue, performs detection using YOLO, 
        # and sends results to results_queue
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                
                # only detections with conf and iou within thresholds are processed
                results = self.model(frame, conf=0.15, iou=0.8, verbose=False)[0]    ### decreased confidence threshold

                inf_msg = self.get_inference_data(results)
                results_queue.put(inf_msg)

    def get_inference_data(self, results):
        # creates visual annotations and extracts bounding box data, class IDs, and confidence scores
        frame = results.orig_img
        annotated_frame = frame.copy()
        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()    ### fixed order for bounding box coordinates
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            c = int(box.cls)
            if self.model.names[c] != "car":
                continue  ### Only process car detections
            
            # cv2.rectangle(image, start_point (top left), end_point (bottom right))
            # draw bounding box
            cv2.rectangle(
                annotated_frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2
            )
            text = f"{self.model.names[c]} | ID: {c} | conf: {box.conf.item():.2f}"
            text_size, _ = cv2.getTextSize(
                text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
            )

            text_width, text_height = text_size
            ### fixed position of text boxes
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1 - 5),
                color=(0, 0, 255),
                thickness=-1,
            )
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
            )

        confs = boxes.conf  # confidence scores
        coords = boxes.xywhn

        """
        results[0]: Original frame (unmodified).
        results[1]: Annotated frame with bounding boxes and labels.
        results[2]: Normalized bounding box coordinates (xywhn).
        results[3]: Class IDs (cls).
        results[4]: Confidence scores (conf).
        """
        inf_msg = (frame, annotated_frame, coords, boxes.cls, confs)

        return inf_msg
