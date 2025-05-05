import numpy as np
from typing import Optional, List, Dict

class ExtendedKalmanFilter:
    """Extended Kalman Filter for smoothing pose estimation keypoints."""
    
    def __init__(self, dt: float = 0.033, process_noise: float = 0.1, measurement_noise: float = 10.0):
        """
        Initialize the EKF for a single keypoint.
        
        Args:
            dt: Time step (seconds, default 1/30 for ~30 FPS)
            process_noise: Process noise covariance (Q matrix)
            measurement_noise: Measurement noise covariance (R matrix)
        """
        # State vector: [x, y, vx, vy] (position and velocity for x and y)
        self.state = np.zeros(4)  # Initial state
        self.P = np.eye(4) * 1.0  # Initial covariance
        
        # Time step
        self.dt = dt
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observes position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # Identity matrix for updates
        self.I = np.eye(4)
    
    def predict(self):
        """Predict the next state based on the motion model."""
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: Optional[np.ndarray], confidence: float = 1.0):
        """
        Update the state with a new measurement.
        
        Args:
            measurement: [x, y] position measurement or None if missing
            confidence: Measurement confidence (0 to 1), adjusts R matrix
        """
        if measurement is None or confidence < 0.1:
            # No measurement, just predict
            return
        
        # Adjust measurement noise based on confidence
        R_adjusted = self.R / max(confidence, 0.1)
        
        # Measurement residual
        z = np.array(measurement)
        y = z - (self.H @ self.state)
        
        # Residual covariance
        S = self.H @ self.P @ self.H.T + R_adjusted
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        self.P = (self.I - K @ self.H) @ self.P
    
    def get_position(self) -> np.ndarray:
        """Return the current estimated position [x, y]."""
        return self.state[:2]

class PoseEKF:
    """Manages EKF for all keypoints in pose estimation."""
    
    def __init__(self, num_keypoints: int = 17, dt: float = 0.033, 
                 process_noise: float = 0.1, measurement_noise: float = 10.0):
        """
        Initialize EKF for all keypoints.
        
        Args:
            num_keypoints: Number of keypoints (default 17 for Thunder model)
            dt: Time step (seconds)
            process_noise: Process noise for EKF
            measurement_noise: Measurement noise for EKF
        """
        self.filters = [
            ExtendedKalmanFilter(dt, process_noise, measurement_noise)
            for _ in range(num_keypoints)
        ]
        self.num_keypoints = num_keypoints
    
    def process(self, keypoints_with_scores: np.ndarray) -> np.ndarray:
        """
        Process keypoints through the EKF.
        
        Args:
            keypoints_with_scores: Shape [1, num_keypoints, 3] with [y, x, confidence]
        
        Returns:
            Smoothed keypoints: Shape [1, num_keypoints, 3] with [y, x, confidence]
        """
        smoothed_keypoints = np.zeros_like(keypoints_with_scores)
        keypoints = np.squeeze(keypoints_with_scores)  # Shape [num_keypoints, 3]
        
        for i, (filter, kp) in enumerate(zip(self.filters, keypoints)):
            y, x, confidence = kp
            
            # Predict next state
            filter.predict()
            
            # Update with measurement if valid
            measurement = np.array([x, y]) if confidence > 0.1 else None
            filter.update(measurement, confidence)
            
            # Get smoothed position
            smoothed_pos = filter.get_position()
            smoothed_keypoints[0, i] = [smoothed_pos[1], smoothed_pos[0], confidence]
        
        return smoothed_keypoints
    
    def reset(self):
        """Reset all filters to initial state."""
        for filter in self.filters:
            filter.state = np.zeros(4)
            filter.P = np.eye(4) * 1.0

def integrate_ekf_with_pose_detector(pose_detector, keypoints_with_scores: np.ndarray) -> np.ndarray:
    """
    Integrate EKF with pose detector output.
    
    Args:
        pose_detector: Instance of PoseDetector from main.py
        keypoints_with_scores: Raw keypoints from pose detector [1, 17, 3]
    
    Returns:
        Smoothed keypoints [1, 17, 3]
    """
    # Initialize EKF if not already done
    if not hasattr(pose_detector, 'ekf'):
        pose_detector.ekf = PoseEKF(num_keypoints=len(pose_detector.keypoint_names))
    
    # Process keypoints through EKF
    smoothed_keypoints = pose_detector.ekf.process(keypoints_with_scores)
    
    return smoothed_keypoints

if __name__ == "__main__":
    # Example usage (for testing purposes)
    import cv2
    from main import PoseDetector, get_square_frame
    
    # Initialize pose detector
    model_path = "singlepose-thunder-tflite-float16.tflite"
    pose_detector = PoseDetector(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        square_frame = get_square_frame(frame)
        keypoints_with_scores, _ = pose_detector.detect_pose(square_frame)
        
        # Apply EKF
        smoothed_keypoints = integrate_ekf_with_pose_detector(pose_detector, keypoints_with_scores)
        
        # Draw results
        frame = pose_detector.draw_keypoints(square_frame, smoothed_keypoints, confidence_threshold=0.2)
        frame = pose_detector.draw_connections(frame, smoothed_keypoints, confidence_threshold=0.2)
        
        cv2.imshow("EKF Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()