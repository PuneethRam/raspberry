import cv2
import numpy as np
import torch
import threading
from queue import Queue
from flask import Flask, request, render_template, jsonify, Response
from ultralytics import solutions
from ultralytics.utils import LOGGER
from datetime import datetime
from flask_cors import CORS
import atexit
import logging
import os
from functools import wraps
import secrets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

LOGGER.setLevel(50)

app = Flask(__name__)

# Security configurations
API_KEY = os.getenv('API_KEY', secrets.token_urlsafe(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
CORS(app, origins=os.getenv('ALLOWED_ORIGINS', '*').split(','))

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🔍 Using device: {device}")

# Configuration
CAM_IDS = ["pi1", "pi2"]
MAX_QUEUE_SIZE = 5
VIDEO_OUTPUT_DIR = os.getenv('VIDEO_OUTPUT_DIR', './videos')
MODEL_PATH = os.getenv('MODEL_PATH', 'best_yolov8_model.pt')
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
VIDEO_FPS = 30

# Create output directory
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# Store latest results & frames
counts = {cam_id: {} for cam_id in CAM_IDS}
latest_frames = {cam_id: None for cam_id in CAM_IDS}
frame_locks = {cam_id: threading.Lock() for cam_id in CAM_IDS}
processing_active = {cam_id: True for cam_id in CAM_IDS}

# Metrics tracking
frame_metrics = {cam_id: {"received": 0, "processed": 0, "errors": 0} for cam_id in CAM_IDS}
metrics_lock = threading.Lock()

# Frame queues
frame_queues = {cam_id: Queue(maxsize=MAX_QUEUE_SIZE) for cam_id in CAM_IDS}

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != API_KEY:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function


# Object counters initialization with error handling
def initialize_counters():
    """Initialize object counters with proper error handling"""
    counters = {}
    for cam_id in CAM_IDS:
        try:
            counters[cam_id] = solutions.ObjectCounter(
                region=[(0, 300), (FRAME_WIDTH, 300)],
                model=MODEL_PATH,
                show=False,
                device=device
            )
            logger.info(f"✅ Initialized counter for {cam_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize counter for {cam_id}: {e}")
            raise
    return counters

object_counters = initialize_counters()

# Video Writers with error handling
def initialize_video_writers():
    """Initialize video writers with proper error handling"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = {}
    for cam_id in CAM_IDS:
        try:
            output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{cam_id}_output.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            if not writer.isOpened():
                raise IOError(f"Failed to open video writer for {cam_id}")
            writers[cam_id] = writer
            logger.info(f"✅ Initialized video writer for {cam_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize video writer for {cam_id}: {e}")
            writers[cam_id] = None
    return writers

video_writers = initialize_video_writers()

def cleanup_resources():
    """Release video writers on shutdown"""
    logger.info("🧹 Cleaning up resources...")
    
    # Stop processing threads
    for cam_id in processing_active:
        processing_active[cam_id] = False
        
    # Send poison pills to queues
    for cam_id in frame_queues:
        try:
            frame_queues[cam_id].put(None, timeout=1)
        except:
            pass
    
    # Release video writers
    for cam_id, writer in video_writers.items():
        if writer is not None:
            try:
                writer.release()
                logger.info(f"✅ Released video writer for {cam_id}")
            except Exception as e:
                logger.error(f"❌ Error releasing writer for {cam_id}: {e}")
    
    # Clean up GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("✅ Cleared GPU cache")

# Register cleanup
atexit.register(cleanup_resources)

def process_frames(cam_id):
    """Background GPU processing thread"""
    logger.info(f"🚀 Started processing thread for {cam_id}")
    
    while processing_active[cam_id]:
        try:
            frame = frame_queues[cam_id].get(timeout=1)
            
            if frame is None:  # Poison pill for shutdown
                break
            
            # Process frame
            results = object_counters[cam_id](frame)
            counts[cam_id] = results.classwise_count
            output_frame = results.plot_im

            # Save to video (non-blocking)
            if video_writers[cam_id] is not None and video_writers[cam_id].isOpened():
                video_writers[cam_id].write(output_frame)

            # Update live preview (thread-safe)
            with frame_locks[cam_id]:
                latest_frames[cam_id] = output_frame
            
            # Update metrics
            with metrics_lock:
                frame_metrics[cam_id]["processed"] += 1

        except Exception as e:
            if processing_active[cam_id]:  # Only log if not shutting down
                logger.error(f"❌ Processing Error {cam_id}: {e}", exc_info=True)
                with metrics_lock:
                    frame_metrics[cam_id]["errors"] += 1
    
    logger.info(f"🛑 Stopped processing thread for {cam_id}")

# Launch processing threads
processing_threads = []
for cam_id in frame_queues:
    thread = threading.Thread(target=process_frames, args=(cam_id,), daemon=True)
    thread.start()
    processing_threads.append(thread)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/<cam_id>', methods=['POST'])
#@require_api_key
def upload_frame(cam_id):
    if cam_id not in frame_queues:
        logger.warning(f"Invalid camera ID: {cam_id}")
        return jsonify({"error": "Invalid camera ID"}), 400

    if not request.data:
        return jsonify({"error": "No frame data received"}), 400

    try:
        # Update metrics
        with metrics_lock:
            frame_metrics[cam_id]["received"] += 1
        
        npimg = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Validate frame decode and dimensions
        if frame is None:
            logger.error(f"Failed to decode frame from {cam_id}")
            return jsonify({"error": "Failed to decode frame"}), 400
        
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Drop old frames (avoid queue lag)
        if frame_queues[cam_id].full():
            try:
                frame_queues[cam_id].get_nowait()
                logger.debug(f"Dropped old frame for {cam_id}")
            except:
                pass

        frame_queues[cam_id].put(frame, timeout=0.1)
        return jsonify({"status": "OK"}), 200
        
    except Exception as e:
        logger.error(f"❌ Upload Error {cam_id}: {e}", exc_info=True)
        with metrics_lock:
            frame_metrics[cam_id]["errors"] += 1
        return jsonify({"error": "Internal server error"}), 500


@app.route('/data')
@require_api_key
def data():
    try:
        response_data = {
            "date": datetime.now().strftime("%d-%m-%Y"),
            "time_stamp": datetime.now().strftime("%H:%M:%S"),
            "waste_data": {
                "Total": {"Incoming": 0, "Outgoing": 0, "Seg_Efficiency": "0%"},
                "Categories": []
            }
        }

        total_in = total_out = 0

        # Get all unique classes
        all_classes = set(counts["pi1"].keys()) | set(counts["pi2"].keys())

        for cls in all_classes:
            incoming = counts["pi1"].get(cls, {}).get("IN", 0)
            outgoing = counts["pi2"].get(cls, {}).get("OUT", 0)
            
            # Fixed: Proper parentheses for efficiency calculation
            seg_eff = ((incoming - outgoing) / incoming * 100) if incoming else 0

            response_data["waste_data"]["Categories"].append({
                "Category": str(cls),
                "Incoming": int(incoming),
                "Outgoing": int(outgoing),
                "Seg_Efficiency": f"{seg_eff:.2f}%"
            })

            total_in += incoming
            total_out += outgoing

        # Fixed: Proper parentheses for total efficiency calculation
        total_eff = ((total_in - total_out) / total_in * 100) if total_in else 0
        response_data["waste_data"]["Total"]["Incoming"] = int(total_in)
        response_data["waste_data"]["Total"]["Outgoing"] = int(total_out)
        response_data["waste_data"]["Total"]["Seg_Efficiency"] = f"{total_eff:.2f}%"

        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"❌ Data Error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


def generate_stream(cam_id):
    """Generate video stream for camera"""
    logger.info(f"Stream started for {cam_id}")
    try:
        while processing_active[cam_id]:
            # Thread-safe frame access
            with frame_locks[cam_id]:
                frame = latest_frames[cam_id]
                
            if frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                               buffer.tobytes() + b'\r\n')
                except Exception as e:
                    logger.error(f"❌ Stream encode error {cam_id}: {e}")
                    break
            else:
                # Wait a bit if no frame available
                threading.Event().wait(0.033)  # ~30 FPS
    finally:
        logger.info(f"Stream ended for {cam_id}")


@app.route('/stream/<cam_id>')
def stream(cam_id):
    if cam_id not in frame_queues:
        return jsonify({"error": "Invalid camera ID"}), 400
        
    return Response(generate_stream(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "device": device,
        "cameras": {},
        "uptime": "N/A"  # Could track with start_time variable
    }
    
    with metrics_lock:
        for cam_id in CAM_IDS:
            health_status["cameras"][cam_id] = {
                "active": processing_active[cam_id],
                "queue_size": frame_queues[cam_id].qsize(),
                "metrics": frame_metrics[cam_id].copy()
            }
    
    return jsonify(health_status)


@app.route('/metrics')
def metrics():
    """Detailed metrics endpoint"""
    with metrics_lock:
        return jsonify({
            "cameras": {cam_id: metrics.copy() for cam_id, metrics in frame_metrics.items()},
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": {
                "allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            } if torch.cuda.is_available() else {}
        })


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large"}), 413


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Validate environment
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        exit(1)
    
    logger.info(f"API Key: {API_KEY[:8]}... (set in environment for production)")
    logger.info(f"Starting server on 0.0.0.0:8080")
    
    try:
        # Use production WSGI server for deployment
        # app.run(host="0.0.0.0", port=8080, threaded=True)  # Development only
        
        # For production, use gunicorn:
        # gunicorn -w 1 -b 0.0.0.0:8080 --timeout 120 --threads 4 app:app
        
        from waitress import serve
        logger.info("Starting production server (waitress)...")
        serve(app, host="0.0.0.0", port=8080, threads=4)
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        cleanup_resources()
