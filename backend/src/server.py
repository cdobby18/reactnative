import cv2
import numpy as np
import tensorflow as tf
import time
import uvicorn
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Optional, List
import asyncio
import base64
from pydantic import BaseModel
import os
from datetime import datetime

# Import your existing pose detector functionality and the new barbell detector
from main import PoseDetector, BarbellDetector, get_square_frame

app = FastAPI(title="Powerlift API")

@app.get("/")
async def root():
    return {"message": "Welcome to the Pose Detection API. Visit /health to check server status."}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the detectors
MODEL_PATH = "../models/singlepose-thunder-tflite-float16.tflite"
BARBELL_MODEL_PATH = "../models/best.pt"
pose_detector = None
barbell_detector = None
confidence_threshold = 0.2

# Global variables
latest_keypoints = None
latest_barbell_results = None
last_detection_time = 0
camera = None
camera_index = 0
current_exercise = "UNKNOWN"

# Recording variables
recording = False
recording_frames = []
output_dir = "../recordings"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define request data models
class FrameRequest(BaseModel):
    image: str
    exercise: Optional[str] = "UNKNOWN"

class RecordingRequest(BaseModel):
    action: str  # "start", "stop"
    exercise: Optional[str] = "UNKNOWN"


@app.on_event("startup")
async def startup_event():
    global pose_detector, barbell_detector, camera, camera_index
    try:
        # Initialize pose detector
        pose_detector = PoseDetector(MODEL_PATH)
        print(f"Pose detector initialized with model: {MODEL_PATH}")
        
        # Initialize barbell detector
        barbell_detector = BarbellDetector(BARBELL_MODEL_PATH)
        print(f"Barbell detector initialized with model: {BARBELL_MODEL_PATH}")
        
        camera = None
        # Try index 2 first (confirmed working), then others as fallback
        for index in [2, 0, 1]:
            print(f"Trying camera index {index}...")
            camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow for better compatibility
            if camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    camera_index = index
                    print(f"Webcam opened successfully with index {camera_index}, resolution: {frame.shape}")
                    # Set reasonable resolution
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
                else:
                    print(f"Failed to read frame at index {index}")
                camera.release()
            else:
                print(f"Camera index {index} not available")
        if camera is None or not camera.isOpened():
            print("Warning: No webcam available. Using mobile frame processing only.")
            camera = None
    except Exception as e:
        print(f"Error during startup: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    global camera
    if camera is not None:
        camera.release()
    print("Released camera resources")

async def process_frames():
    global latest_keypoints, latest_barbell_results, last_detection_time
    
    while True:
        if camera is None or not camera.isOpened():
            await asyncio.sleep(0.1)
            continue
            
        ret, frame = camera.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue
        
        square_frame = get_square_frame(frame)
        
        try:
            # Detect pose
            keypoints_with_scores, _ = pose_detector.detect_pose(square_frame)
            latest_keypoints = keypoints_with_scores.tolist()
            
            # Detect barbell
            if barbell_detector is not None:
                barbell_results = barbell_detector.detect(square_frame)
                latest_barbell_results = barbell_results
            
            last_detection_time = time.time()
        except Exception as e:
            print(f"Error in detection: {str(e)}")
        
        await asyncio.sleep(0.01)

@app.post("/process_frame")
async def process_frame(request: FrameRequest):
    """Process a frame from a mobile device camera"""
    global pose_detector, barbell_detector, current_exercise, recording, recording_frames
    
    if request.exercise:
        current_exercise = request.exercise

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not decode image"}
            )
        
        # Process image with pose detector
        keypoints_with_scores, _ = pose_detector.detect_pose(img)
        
        # Process image with barbell detector if available
        barbell_results = None
        if barbell_detector is not None:
            barbell_results = barbell_detector.detect(img)
        
        # Draw keypoints and connections for pose
        img = pose_detector.draw_keypoints(img, keypoints_with_scores, confidence_threshold=0.2)
        img = pose_detector.draw_connections(img, keypoints_with_scores, confidence_threshold=0.2)
        
        # Draw barbell detections if available
        if barbell_results is not None:
            img = barbell_detector.draw_on_frame(img, barbell_results)
        
        # Add exercise name on the image
        cv2.putText(
            img, 
            f"Exercise: {current_exercise}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )
        
        # Add recording indicator if recording
        if recording:
            cv2.putText(
                img, 
                "RECORDING", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255), 
                2
            )
            # Add frame to recording
            recording_frames.append(img.copy())
        
        # Get keypoints data
        keypoints = []
        shaped = np.squeeze(keypoints_with_scores)
        for i, (y, x, confidence) in enumerate(shaped):
            if confidence > confidence_threshold:
                keypoints.append({
                    "id": i,
                    "name": pose_detector.keypoint_names[i],
                    "position": {"x": float(x), "y": float(y)},
                    "confidence": float(confidence)
                })
        
        # Get exercise analysis with barbell information if available
        feedback = ""
        try:
            if hasattr(pose_detector, 'analyze_with_barbell') and barbell_results is not None:
                analysis = pose_detector.analyze_with_barbell(keypoints_with_scores, barbell_results, current_exercise)
                feedback = analysis.get("feedback", "")
            elif hasattr(pose_detector, 'analyze_exercise'):
                analysis = pose_detector.analyze_exercise(keypoints_with_scores, current_exercise)
                feedback = analysis.get("feedback", "")
        except Exception as analysis_error:
            print(f"Error in exercise analysis: {analysis_error}")
            
        # Create barbell data for response
        barbell_data = None
        if barbell_results is not None:
            barbell_data = {
                "detected": len(barbell_results.get('boxes', [])) > 0 or barbell_results.get('keypoints') is not None,
                "boxes": barbell_results.get('boxes', [])
            }
            
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_encoded = base64.b64encode(buffer).decode('utf-8')
        
        # Return processed image and analysis
        return {
            "image": img_encoded,
            "keypoints": keypoints,
            "barbell": barbell_data,
            "feedback": feedback,
            "recording": recording
        }
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing frame: {str(e)}"}
        )

@app.post("/recording")
async def control_recording(request: RecordingRequest):
    """Start or stop recording frames with pose detection"""
    global recording, recording_frames, current_exercise
    
    if request.exercise:
        current_exercise = request.exercise
    
    if request.action == "start":
        recording = True
        recording_frames = []
        return {"status": "ok", "recording": True}
    
    elif request.action == "stop":
        recording = False
        
        # Save the recorded frames as a video if there are any
        if recording_frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/{current_exercise}_{timestamp}.mp4"
            
            if len(recording_frames) > 0:
                height, width, _ = recording_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, 15.0, (width, height))
                
                for frame in recording_frames:
                    out.write(frame)
                
                out.release()
                
                return {
                    "status": "ok", 
                    "recording": False, 
                    "message": f"Video saved as {filename}",
                    "frames": len(recording_frames),
                    "path": filename
                }
            
        return {"status": "ok", "recording": False, "message": "No frames recorded"}
    
    return {"status": "error", "message": "Invalid action"}

@app.get("/video_feed")
async def video_feed(front: bool = False, t: Optional[str] = None, exercise: Optional[str] = None):
    global camera, current_exercise
    
    if exercise:
        current_exercise = exercise
    
    if front and camera_index != 1:
        try:
            await switch_camera(1, front=True)
        except Exception as e:
            print(f"Failed to switch to front camera: {e}")
    
    async def generate_frames():
        while True:
            try:
                if camera is None or not camera.isOpened():
                    error_frame = np.zeros((320, 320, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "No Camera Available", (50, 160), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    _, jpeg = cv2.imencode('.jpg', error_frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    await asyncio.sleep(0.5)
                    continue
                
                ret, frame = camera.read()
                if not ret:
                    await asyncio.sleep(0.1)
                    continue
                # Process the frame
                square_frame = get_square_frame(frame)
                square_frame = cv2.resize(square_frame, (320, 320))
                
                # Draw pose detection
                if latest_keypoints is not None and pose_detector is not None:
                    keypoints_array = np.array(latest_keypoints)
                    square_frame = pose_detector.draw_keypoints(square_frame, keypoints_array, confidence_threshold)
                    square_frame = pose_detector.draw_connections(square_frame, keypoints_array, confidence_threshold)
                
                # Draw barbell detection
                if latest_barbell_results is not None and barbell_detector is not None:
                    square_frame = barbell_detector.draw_on_frame(square_frame, latest_barbell_results)
                
                # Add exercise name
                cv2.putText(
                    square_frame, 
                    f"Exercise: {current_exercise}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    1
                )
                
                # Add recording indicator
                if recording:
                    cv2.putText(
                        square_frame, 
                        "RECORDING", 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 0, 255), 
                        1
                    )
                
                # Add barbell status
                if latest_barbell_results is not None:
                    barbell_detected = (len(latest_barbell_results.get('boxes', [])) > 0 or 
                                       latest_barbell_results.get('keypoints') is not None)
                    status_text = "Barbell: Detected" if barbell_detected else "Barbell: Not Detected"
                    cv2.putText(
                        square_frame, 
                        status_text, 
                        (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 255), 
                        1
                    )
                
                _, jpeg = cv2.imencode('.jpg', square_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = jpeg.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
                       frame_bytes + b'\r\n')
                
                await asyncio.sleep(0.033)  # ~30 fps
            
            except Exception as e:
                print(f"Error in generate_frames: {e}")
                await asyncio.sleep(0.1)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/switch_camera")
async def switch_camera(camera_id: int = 0, front: bool = False):
    """Switch between cameras"""
    global camera, camera_index
    
    try:
        if camera is not None:
            camera.release()
        
        if front:
            success = False
            for index in [1, 2, 0]:  # Try front camera indexes
                camera = cv2.VideoCapture(index)
                if camera.isOpened():
                    camera_index = index
                    success = True
                    print(f"Switched to front camera with index {index}")
                    break
            
            if not success:
                camera = cv2.VideoCapture(0)
                camera_index = 0
        else:
            camera = cv2.VideoCapture(camera_id)
            camera_index = camera_id
        
        return {"status": "ok", "camera_index": camera_index}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    camera_status = False
    if camera is not None and camera.isOpened():
        ret, _ = camera.read()
        camera_status = ret
    
    barbell_detector_status = barbell_detector is not None
    
    return {
        "status": "ok", 
        "camera": camera_status,
        "camera_index": camera_index,
        "pose_model": pose_detector is not None,
        "barbell_model": barbell_detector_status,
        "recording": recording,
        "server_time": time.time(),
        "current_exercise": current_exercise
    }

@app.get("/pose_data")
async def pose_data() -> Dict:
    """Returns comprehensive data including pose keypoints and barbell detection"""
    if latest_keypoints is None:
        return {"error": "No pose data available yet"}
    
    # Process pose keypoints
    keypoints = []
    if latest_keypoints:
        shaped = np.squeeze(latest_keypoints)
        for i, (y, x, confidence) in enumerate(shaped):
            if confidence > confidence_threshold:
                keypoints.append({
                    "id": i,
                    "name": pose_detector.keypoint_names[i] if pose_detector else f"kp{i}",
                    "position": {"x": float(x), "y": float(y)},
                    "confidence": float(confidence)
                })
    
    # Process barbell data
    barbell_data = {
        "detected": False,
        "boxes": []
    }
    
    if latest_barbell_results is not None:
        barbell_detected = (len(latest_barbell_results.get('boxes', [])) > 0 or 
                           latest_barbell_results.get('keypoints') is not None)
        barbell_data["detected"] = barbell_detected
        barbell_data["boxes"] = latest_barbell_results.get('boxes', [])
        
        if latest_barbell_results.get('keypoints') is not None:
            barbell_data["keypoints"] = []
            for i, kp in enumerate(latest_barbell_results['keypoints']):
                if kp is not None:
                    barbell_data["keypoints"].append({
                        "id": i,
                        "position": {"x": float(kp[0]), "y": float(kp[1])}
                    })
    
    # Get exercise analysis with barbell information if available
    feedback = ""
    analysis_result = {}
    try:
        if pose_detector and hasattr(pose_detector, 'analyze_with_barbell') and latest_barbell_results is not None:
            analysis_result = pose_detector.analyze_with_barbell(
                np.array(latest_keypoints), 
                latest_barbell_results, 
                current_exercise
            )
            feedback = analysis_result.get("feedback", "")
        elif pose_detector and hasattr(pose_detector, 'analyze_exercise'):
            analysis_result = pose_detector.analyze_exercise(np.array(latest_keypoints), current_exercise)
            feedback = analysis_result.get("feedback", "")
    except Exception as analysis_error:
        print(f"Error in exercise analysis: {analysis_error}")
    
    return {
        "pose": {
            "keypoints": keypoints
        },
        "barbell": barbell_data,
        "analysis": analysis_result,
        "feedback": feedback,
        "exercise": current_exercise,
        "timestamp": last_detection_time,
        "recording": recording
    }

@app.get("/set_exercise/{exercise}")
async def set_exercise(exercise: str):
    """Set the current exercise type"""
    global current_exercise
    current_exercise = exercise
    return {"status": "ok", "exercise": current_exercise}

@app.get("/recordings")
async def list_recordings():
    """List all available recordings"""
    recordings = []
    
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith(".mp4"):
                file_path = os.path.join(output_dir, file)
                file_stats = os.stat(file_path)
                recordings.append({
                    "filename": file,
                    "path": file_path,
                    "size_bytes": file_stats.st_size,
                    "created": file_stats.st_ctime,
                    "exercise": file.split("_")[0] if "_" in file else "unknown"
                })
    
    return {"recordings": recordings}

# Start the background task for continuous frame processing
@app.on_event("startup")
async def start_frame_processing():
    asyncio.create_task(process_frames())

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)