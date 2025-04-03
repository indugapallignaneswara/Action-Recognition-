# app.py

from flask import Flask, request, jsonify, render_template
from flask import send_file
import os
from detectors.detector import Detector
from werkzeug.utils import secure_filename
from utils.utils import draw_boxes_on_video

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    yolo_version = request.form.get('yolo_version', 'v8')
    conf_thres = request.form.get('conf_thres', default=0.3, type=float)

    filename = secure_filename(video.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(video_path)

    try:
        detector = Detector(conf_thres=conf_thres)
        results = detector.detect_and_track(input_video=video_path, yolo_version=yolo_version)

        model_name = filename.split('.')[0]
        resized_video_path = f"outputs/{model_name}_resized.mp4"
        boxed_video_path = f"outputs/{model_name}_boxed.mp4"

        final_path = os.path.join('static', os.path.basename(boxed_video_path))
        boxed_video_path = draw_boxes_on_video(resized_video_path, results)
        os.replace(boxed_video_path, final_path)

        return jsonify({
            'message': 'Detection complete',
            'video_url': f"/static/{os.path.basename(final_path)}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
