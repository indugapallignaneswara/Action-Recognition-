from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import cv2
from detectors.detector import Detector
from utils.utils import draw_boxes_on_video
from classifier.classifier import feature_extractor, load_trained_transformer_model, predict_action
import json, shutil

app = Flask(__name__, template_folder='templates')

# Constants
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
MAX_SEQ_LENGTH = 32

# Ensure required directories exist
def ensure_session_dirs(session_id):
    session_dirs = [
        f'uploads/session{session_id}',
        f'outputs/session{session_id}/crops',
        f'static/session{session_id}'
    ]
    for d in session_dirs:
        os.makedirs(d, exist_ok=True)

# Get next available session ID
def get_next_session_id():
    existing = [int(d.replace('session', '')) for d in os.listdir('uploads') if d.startswith('session')]
    return max(existing, default=0) + 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video selected'}), 400

        file_ext = os.path.splitext(video_file.filename)[1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            return jsonify({'error': f'Unsupported format. Supported: {", ".join(SUPPORTED_FORMATS)}'}), 400

        yolo_version = request.form.get('yolo_version', 'v8')
        conf_thres = request.form.get('conf_thres', '0.3')

        try:
            conf_thres = float(conf_thres)
            if not (0 < conf_thres <= 1):
                return jsonify({'error': 'Confidence threshold must be between 0 and 1'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid confidence threshold value'}), 400

        session_id = get_next_session_id()
        ensure_session_dirs(session_id)

        original_name = os.path.splitext(video_file.filename)[0]
        unique_id = uuid.uuid4().hex
        unique_filename = f"{unique_id}_{original_name}{file_ext}"
        upload_path = f"uploads/session{session_id}/{unique_filename}"
        video_file.save(upload_path)

        # 🧩 Patch: Save original input video to crops/ as well
        # Ensure crops directory exists
        os.makedirs(f"outputs/session{session_id}/crops", exist_ok=True)
        # Define target path
        temp_crop_copy = f"outputs/session{session_id}/crops/{unique_filename}"
        # Copy the uploaded video
        shutil.copy(upload_path, temp_crop_copy)

        detector = Detector(conf_thres=conf_thres, device='cuda')
        model_name = f"{unique_id}_{original_name}"

        # Force all output paths to session output folder
        resized_path = f"outputs/session{session_id}/{model_name}_resized.mp4"
        result_path = f"outputs/session{session_id}/{model_name}_result.mp4"
        json_output_path = f"outputs/session{session_id}/{model_name}_detections.json"

        detection_results = detector.detect_and_track(upload_path, yolo_version=yolo_version)
        draw_boxes_on_video(resized_path, detection_results, result_path)

        with open(json_output_path, 'w') as f:
            json.dump(detection_results, f, indent=2)

        # 🧠 Classification step on all cropped videos in crops directory
        model = load_trained_transformer_model()
        classification_results = []
        crop_dir = f"outputs/session{session_id}/crops"

        print("\n🔍 ---- CLASSIFICATION START ----")
        print(f"📁 Looking in: {crop_dir}")
        print(f"📄 Files found: {os.listdir(crop_dir)}\n")

        if not os.path.exists(crop_dir):
            print(f"❌ ERROR: Crop directory {crop_dir} does not exist.")
        else:
            for crop_file in os.listdir(crop_dir):
                print("enter for")
                if crop_file.endswith(".mp4") or crop_file.endswith(".avi") or crop_file.endswith(".mov"):
                    print("entered if")
                    crop_full_path = os.path.join(crop_dir, crop_file)
                    print(f"🧪 Classifying: {crop_file}")
                    try:
                        predicted_class = predict_action(crop_full_path, model, label_enc)
                        print(f"✅ Predicted: {predicted_class}")
                    except Exception as e:
                        print(f"❌ Failed to classify {crop_file}: {e}")
                        continue

                    class_result = {
                        'session_id': session_id,
                        'video': crop_file,
                        'crop_path': crop_full_path,
                        'predicted_class': predicted_class,
                        # 'confidence': confidence
                    }

                    class_json_path = f"outputs/session{session_id}/{crop_file.replace('.mp4', '_classification.json')}"
                    try:
                        with open(class_json_path, 'w') as f:
                            json.dump(class_result, f, indent=2)
                        print(f"💾 Saved classification result to: {class_json_path}\n")
                    except Exception as e:
                        print(f"❌ Failed to save JSON: {e}")

                    classification_results.append(class_result)
                else:
                    print("exited")

        print("✅ CLASSIFICATION COMPLETE")
        print(f"📦 Total classified videos: {len(classification_results)}")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'video_url': f'/output/session{session_id}/{model_name}_result.mp4',
            'json_url': f'/output/session{session_id}/{model_name}_detections.json',
            'classification': classification_results
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory('outputs', filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory('uploads', filename)

@app.route('/cleanup', methods=['POST'])
def cleanup():
    return jsonify({'success': True, 'message': 'Cleanup placeholder'})

@app.route('/gallery')
def gallery():
    try:
        videos = []
        for root, _, files in os.walk('outputs'):
            for file in files:
                if file.endswith('_result.mp4'):
                    rel_path = os.path.join(root, file).replace('outputs/', '')
                    videos.append({'filename': file, 'url': f'/output/{rel_path}'})
        return render_template('gallery.html', videos=videos)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
