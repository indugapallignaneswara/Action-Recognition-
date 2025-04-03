import os
import cv2

def draw_boxes_on_video(resized_video_path, detection_results):
    cap = cv2.VideoCapture(resized_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.basename(resized_video_path).replace("_resized", "_boxed")
    output_path = os.path.join("outputs", base_name)

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
