from ultralytics import YOLO
import cv2
import numpy as np


official_model = YOLO('yolov8s-pose.pt')  
custom_model = YOLO('best.pt')          

video_path = "squat.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# confidence threshold for keypoints
CONF_THRESH = 0.3
SMOOTHING_FACTOR = 0.8  
prev_keypoints = None

def filter_and_interpolate_keypoints(keypoints, confidences, prev_kp):
    """ Filters low-confidence keypoints and interpolates missing ones. """
    filtered_kps = []
    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
        if conf < CONF_THRESH:
            if prev_kp is not None:
                kp = prev_kp[i]
            else:
                kp = None  
        filtered_kps.append(kp)
    return filtered_kps

def smooth_keypoints(current_kp, prev_kp):
    """ Applies Exponential Moving Average (EMA) for smoothing. """
    if prev_kp is None:
        return current_kp  
    
    smoothed_kps = []
    for cur, prev in zip(current_kp, prev_kp):
        if cur is None:
            smoothed_kps.append(prev)
        else:
            smoothed_kps.append(
                (SMOOTHING_FACTOR * np.array(prev)) + ((1 - SMOOTHING_FACTOR) * np.array(cur))
            )
    return smoothed_kps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    official_results = official_model(frame, conf=0.5)
    plotted_frame = official_results[0].plot(boxes=False)  

    custom_results = custom_model(frame, conf=0.8)

    if custom_results[0].keypoints is not None:
        raw_keypoints = custom_results[0].keypoints.xy.cpu().numpy()
        confidences = custom_results[0].keypoints.conf.cpu().numpy()
    else:
        raw_keypoints = None
        confidences = None

    if raw_keypoints is not None:
        filtered_keypoints = filter_and_interpolate_keypoints(raw_keypoints[0], confidences[0], prev_keypoints)

        smoothed_keypoints = smooth_keypoints(filtered_keypoints, prev_keypoints)
        prev_keypoints = smoothed_keypoints 

        for kp in smoothed_keypoints:
            if kp is not None:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(plotted_frame, (x, y), 5, (0, 0, 255), -1)  


    out.write(plotted_frame)
    
    #cv2.imshow('Ensemble Model Prediction', plotted_frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
out.release()
cv2.destroyAllWindows()
