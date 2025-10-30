import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, jsonify, Response
from ultralytics import solutions
from ultralytics.utils import LOGGER
import multiprocessing
from datetime import datetime

LOGGER.setLevel(50)

app = Flask(__name__)

# CUDA or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Using device: {device}")

# Shared dictionary to store counts and latest frames
manager = multiprocessing.Manager()
counts = manager.dict({"pi1": {}, "pi2": {}})
latest_frames = manager.dict({"pi1": None, "pi2": None})

# Object counters
object_counters = {
    "pi1": solutions.ObjectCounter(
        region=[(0, 300), (640, 300)],
        model="best_yolov8_model.pt",
        show=False,
        device=device
    ),
    "pi2": solutions.ObjectCounter(
        region=[(0, 300), (640, 300)],
        model="best_yolov8_model.pt",
        show=False,
        device=device
    )
}

# Video writer setup
video_writers = {}
frame_size = (640, 480)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
for cam_id in object_counters.keys():
    output_file = f"{cam_id}_output.mp4"
    video_writers[cam_id] = cv2.VideoWriter(output_file, fourcc, 30, frame_size)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/<cam_id>', methods=['POST'])
def upload_frame(cam_id):
    if cam_id not in object_counters:
        return "Invalid camera ID", 400

    try:
        npimg = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        results = object_counters[cam_id](frame)
        counts[cam_id] = results.classwise_count

        # Save to video
        video_writers[cam_id].write(results.plot_im)

        # Store for streaming
        latest_frames[cam_id] = results.plot_im

        return "OK", 200
    except Exception as e:
        print(f"❌ Error processing frame from {cam_id}: {e}")
        return "Error", 500


@app.route('/data')
def data():
    response_data = {
        "date": datetime.now().strftime("%d-%m-%Y"),
        "time_stamp": datetime.now().strftime("%H:%M"),
        "waste_data": {
            "Total": {
                "Incoming": 0,
                "Outgoing": 0,
                "Seg_Efficiency": "0%",
                "Least_efficient_category": "",
                "Most_efficient_category": ""
            },
            "Categories": []
        }
    }

    total_incoming = 0
    total_outgoing = 0
    efficiency_map = {}

    for cls, pi1_data in counts["pi1"].items():
        incoming = pi1_data.get("IN", 0)
        outgoing = counts["pi2"].get(cls, {}).get("IN", 0)

        seg_eff = (outgoing / incoming * 100) if incoming > 0 else 0
        efficiency_str = f"{seg_eff:.2f}%"

        response_data["waste_data"]["Categories"].append({
            "Category": cls,
            "Incoming": incoming,
            "Outgoing": outgoing,
            "Seg_Efficiency": efficiency_str
        })

        total_incoming += incoming
        total_outgoing += outgoing
        efficiency_map[cls] = seg_eff

    # Compute total efficiency
    total_eff = (total_outgoing / total_incoming * 100) if total_incoming > 0 else 0
    response_data["waste_data"]["Total"]["Incoming"] = total_incoming
    response_data["waste_data"]["Total"]["Outgoing"] = total_outgoing
    response_data["waste_data"]["Total"]["Seg_Efficiency"] = f"{total_eff:.2f}%"

    # Find least/most efficient categories
    if efficiency_map:
        least = min(efficiency_map, key=efficiency_map.get)
        most = max(efficiency_map, key=efficiency_map.get)
        response_data["waste_data"]["Total"]["Least_efficient_category"] = least
        response_data["waste_data"]["Total"]["Most_efficient_category"] = most

    return jsonify(response_data)

# MJPEG video stream generator
def generate_stream(cam_id):
    while True:
        frame = latest_frames.get(cam_id)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # If no frame available yet, wait a bit
            import time
            time.sleep(0.1)

@app.route('/stream/<cam_id>')
def stream_video(cam_id):
    if cam_id not in object_counters:
        return "Invalid camera ID", 400
    return Response(generate_stream(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def release_all_video_writers():
    for vw in video_writers.values():
        vw.release()
    print("✅ Video writers released.")

def main():
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        release_all_video_writers()

if __name__ == "__main__":
    main()

