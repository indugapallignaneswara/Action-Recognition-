import os
import cv2
from collections import defaultdict
import numpy as np
import os

import cv2
import numpy as np

def load_single_video(video_path, max_frames=32, resize=(224, 224)):
    import cv2
    import numpy as np

    print(f"\n📥 Loading video: {video_path}")
    frames = []
    cap = cv2.VideoCapture(video_path)

    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ End of video or unreadable frame at count {len(frames)}")
                break

            try:
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, ::-1]  # BGR to RGB
                frame = frame / 255.0
                frames.append(frame)
            except Exception as e:
                print(f"❌ Error processing frame {len(frames)}: {e}")
                continue

    finally:
        cap.release()

    print(f"📦 Frames loaded: {len(frames)}")

    if len(frames) == 0:
        print(f"❌ Skipping video {video_path} — no usable frames.")
        raise ValueError(f"Video {video_path} is empty or unreadable.")

    try:
        frames = np.stack(frames, axis=0)
    except Exception as e:
        print(f"❌ Error stacking frames: {e}")
        print(f"🧪 frames is type: {type(frames)}; first shape: {getattr(frames[0], 'shape', 'N/A')}")
        raise

    if frames.shape[0] < max_frames:
        pad_len = max_frames - frames.shape[0]
        pad = np.zeros((pad_len, resize[1], resize[0], 3), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)

    print(f"✅ Final video tensor shape: {frames.shape}")
    return frames




def get_next_session_id(base_dir='uploads'):
    existing = [d for d in os.listdir(base_dir) if d.startswith('session')]
    numbers = [int(name.replace('session', '')) for name in existing if name.replace('session', '').isdigit()]
    next_id = max(numbers + [0]) + 1
    return f'session{next_id}'

def draw_boxes_on_video(resized_video_path, detection_results, output_path):
    cap = cv2.VideoCapture(resized_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.basename(resized_video_path).replace("_resized", "_boxed")
    # output_path = os.path.join("outputs", base_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_detections = [
            d for d in detection_results if d['frame'] == frame_idx
        ]

        for det in current_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            tid = det['track_id']
            conf = det['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'ID: {tid}, Conf: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[✓] Annotated video saved to: {output_path}")
    return output_path

def crop_individuals_from_video(video_path, detection_results, output_dir='outputs/crops', min_frames=5):
    os.makedirs(output_dir, exist_ok=True)

    # Organize detections by frame
    frame_detections = defaultdict(list)
    for det in detection_results:
        frame_detections[det['frame']].append(det)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    writers = {}  # track_id -> cv2.VideoWriter
    track_frame_counts = defaultdict(int)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_detections:
            for det in frame_detections[frame_idx]:
                tid = det['track_id']
                x1, y1, x2, y2 = map(int, det['bbox'])

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Initialize writer if not done
                if tid not in writers:
                    crop_h, crop_w = crop.shape[:2]
                    writer_path = os.path.join(output_dir, f'person_{tid}.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writers[tid] = cv2.VideoWriter(writer_path, fourcc, fps, (crop_w, crop_h))

                writers[tid].write(crop)
                track_frame_counts[tid] += 1

        frame_idx += 1

    cap.release()

    # Release and remove short clips
    for tid, writer in writers.items():
        writer.release()
        if track_frame_counts[tid] < min_frames:
            path = os.path.join(output_dir, f'person_{tid}.mp4')
            os.remove(path)
            print(f"[ℹ️] Removed short clip for person_{tid} (<{min_frames} frames)")

    valid_tracks = [tid for tid, count in track_frame_counts.items() if count >= min_frames]
    print(f"[✓] Saved {len(valid_tracks)} cropped person videos to: {output_dir}")
    return len(valid_tracks)
