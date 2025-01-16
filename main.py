from readers import VideoReader
from detectors import VehicleDetector
from tracers import CarTracer
import threading
import queue

video_path = "./videos/drone_footage.mp4"
model_path = "./model/road_v2.pt"
bg_path = "./bg.png"

if __name__ == "__main__":
    frame_queue = queue.Queue()     # share frames between video streamer and detector     
    results_queue = queue.Queue()   # share detection results between detector and tracer

    streamer = VideoReader(video_path)  # stream frames from video file
    video_width, video_height = streamer.get_video_properties() # frame dimentsions

    detector = VehicleDetector(model_path)  # detect vehicles using YOLO model
    tracer = CarTracer(bg_path, video_width, video_height)  # traces detected vehicles + visualizations

    # enables simultaneous streaming and detection
    streamer_thread = threading.Thread(target=streamer.stream, args=(frame_queue,))
    detection_thread = threading.Thread(
        target=detector.detect,
        args=(
            frame_queue,
            results_queue,
        ),
    )
    streamer_thread.start()
    detection_thread.start()

    tracer.trace(results_queue)

    streamer_thread.join()
    frame_queue.put(None)
    detection_thread.join()
    results_queue.put(None)
