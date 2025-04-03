# detectors/detector.py

import os
import cv2
import torch
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.utils import draw_boxes_on_video

class Detector:
    def __init__(self, conf_thres=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_thres = conf_thres if isinstance(conf_thres, (float, int)) and 0 < conf_thres <= 1 else 0.3
        self.tracker = DeepSort(max_age=30)
        self.model = None  # Lazy load in detect_and_track

        # # Check if model file exists
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"[ERROR] Model file not found: {model_path}")

        # self.model = YOLO(model_path).to(self.device)

        # Validate threshold
        if conf_thres is None or not isinstance(conf_thres, (float, int)) or not (0 < conf_thres <= 1):
            print(f"[INFO] Invalid or missing threshold. Using default: 0.3")
            conf_thres = 0.3
        elif isinstance(conf_thres, int) and conf_thres > 1:
            conf_thres = conf_thres / 100.0
        elif not (0 < conf_thres <= 1):
            print(f"[INFO] Invalid threshold '{conf_thres}', using default: 0.3")
            conf_thres = 0.3

        self.conf_thres = conf_thres

        # DeepSORT tracker
        self.tracker = DeepSort(max_age=30)

    def preprocess_video(self, input_path, output_path, target_size=(640, 640), frame_skip=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"[ERROR] Input video file not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if total_frames == 0 or fps == 0:
            raise ValueError("[ERROR] Unable to read video properties.")

        duration_secs = total_frames / fps

        # 🧠 Calculate dynamic frame skip if not provided
        if frame_skip is None:
            target_frames = 12
            frame_skip = max(1, total_frames // target_frames)
            print(f"[INFO] Video duration = {round(duration_secs, 2)}s, Total frames = {total_frames}, "
                f"Auto frame_skip = {frame_skip}")

        new_fps = fps // frame_skip
        if new_fps == 0:  # Ensure valid fps
            new_fps = 1

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            new_fps,
            target_size
        )

        frames = []
        frame_count = 0
        timestamps = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                resized = cv2.resize(frame, target_size)
                out.write(resized)
                frames.append(resized)
                timestamps.append(frame_count / fps)
            frame_count += 1

        cap.release()
        out.release()
        return frames, timestamps, new_fps


    def detect_and_track(self, input_video, yolo_version="v8"):
        # Validate video input
        if not input_video or not isinstance(input_video, str):
            raise ValueError("[ERROR] Invalid input video path.")

        if not os.path.exists(input_video):
            raise FileNotFoundError(f"[ERROR] Video file not found: {input_video}")

        # Validate YOLO version
        if yolo_version not in ["v8", "v11"]:
            raise ValueError("[ERROR] YOLO version must be 'v8' or 'v11'.")

        # Auto-set model path
        if yolo_version == "v8":
            model_path = "detectors/yolov8m.pt"
        elif yolo_version == "v11":
            model_path = "detectors/yolo11x.pt"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] YOLO model file '{model_path}' not found. "
                                    f"Please ensure it's placed in the working directory.")

        # Load YOLO model based on version
        self.model = YOLO(model_path).to(self.device)

        model_name = os.path.basename(input_video).split('.')[0]
        resized_video_path = f"outputs/{model_name}_resized.mp4"

        frames, timestamps, fps = self.preprocess_video(input_video, resized_video_path)

        detection_results = []
        printf(f"model used for inference", model_name)
        for idx, frame in enumerate(frames):
            results = self.model(frame, conf=self.conf_thres)[0]
            detections = []

            for box in results.boxes:
                cls = int(box.cls.item())
                if cls != 0:  # Only detect 'person'
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

            tracks = self.tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            conf = track.get_det_conf() or 0.0

            if conf < self.conf_thres:
                continue  # ❌ skip low-confidence tracks

            x1, y1, x2, y2 = map(int, ltrb)
            timestamp_value = timestamps[idx] if idx < len(timestamps) and timestamps[idx] is not None else 0.0
            detection_results.append({
                'track_id': track_id,
                'frame': idx,
                'timestamp': round(timestamp_value, 2),
                'bbox': [x1, y1, x2, y2],
                'confidence': round(conf, 4)
            })


        # Save to JSON
        output_json = f"outputs/{model_name}_detections.json"
        with open(output_json, 'w') as f:
            json.dump(detection_results, f, indent=2)

        print(f"[✓] Detection + tracking complete. Results saved to: {output_json}")
        return detection_results
