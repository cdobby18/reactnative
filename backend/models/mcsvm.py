import numpy as np
from typing import Dict, List, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import os

class FormClassifier:
    """Multi-Class SVM for classifying exercise form based on pose keypoints."""
    
    def __init__(self, model_dir: str = "models", exercise_type: str = "SQUAT"):
        """
        Initialize the form classifier.
        
        Args:
            model_dir: Directory to save/load trained models
            exercise_type: Type of exercise ("SQUAT", "DEADLIFT", "BENCH")
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.exercise_type = exercise_type.upper()
        self.classes = ["Good Form", "Needs Improvement", "Poor Form"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Initialize SVM with probability estimates
        self.svm = SVC(kernel='rbf', probability=True, decision_function_shape='ovr')
        self.scaler = StandardScaler()
        
        # Model file path
        self.model_path = os.path.join(model_dir, f"svm_{self.exercise_type.lower()}.pkl")
        
        # Load model if it exists
        self.load_model()
    
    def extract_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract features from keypoints for classification.
        
        Args:
            keypoints: Shape [1, 17, 3] with [y, x, confidence]
        
        Returns:
            Feature vector
        """
        keypoints = np.squeeze(keypoints)  # Shape [17, 3]
        features = []
        
        # Helper function to calculate angle between three points
        def calculate_angle(p1, p2, p3):
            v1 = p1[:2] - p2[:2]
            v2 = p3[:2] - p2[:2]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.degrees(np.arccos(cos_theta))
        
        # Helper function to calculate normalized distance
        def calculate_distance(p1, p2):
            return np.linalg.norm(p1[:2] - p2[:2])
        
        if self.exercise_type == "SQUAT":
            # Keypoints: left_hip (11), left_knee (13), left_ankle (15), right_hip (12), right_knee (14), right_ankle (16)
            # Features: knee angles, hip angles, knee-to-hip distance
            left_knee_angle = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
            right_knee_angle = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
            left_hip_angle = calculate_angle(keypoints[5], keypoints[11], keypoints[13])
            right_hip_angle = calculate_angle(keypoints[6], keypoints[12], keypoints[14])
            hip_knee_dist = calculate_distance(keypoints[11], keypoints[13]) / calculate_distance(keypoints[11], keypoints[12])
            features.extend([left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle, hip_knee_dist])
        
        elif self.exercise_type == "DEADLIFT":
            # Keypoints: left_shoulder (5), left_hip (11), left_knee (13), right_shoulder (6), right_hip (12), right_knee (14)
            # Features: back angle, hip angle, shoulder-to-hip alignment
            back_angle_left = calculate_angle(keypoints[5], keypoints[11], keypoints[13])
            back_angle_right = calculate_angle(keypoints[6], keypoints[12], keypoints[14])
            hip_angle_left = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
            hip_angle_right = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
            shoulder_hip_dist = calculate_distance(keypoints[5], keypoints[11]) / calculate_distance(keypoints[5], keypoints[6])
            features.extend([back_angle_left, back_angle_right, hip_angle_left, hip_angle_right, shoulder_hip_dist])
        
        elif self.exercise_type == "BENCH":
            # Keypoints: left_shoulder (5), left_elbow (7), left_wrist (9), right_shoulder (6), right_elbow (8), right_wrist (10)
            # Features: elbow angles, wrist alignment
            left_elbow_angle = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
            right_elbow_angle = calculate_angle(keypoints[6], keypoints[8], keypoints[10])
            shoulder_wrist_dist = calculate_distance(keypoints[5], keypoints[9]) / calculate_distance(keypoints[5], keypoints[6])
            features.extend([left_elbow_angle, right_elbow_angle, shoulder_wrist_dist])
        
        return np.array(features)
    
    def train(self, keypoints_list: List[np.ndarray], labels: List[str]):
        """
        Train the SVM model with labeled keypoint data.
        
        Args:
            keypoints_list: List of keypoint arrays [1, 17, 3]
            labels: List of labels ("Good Form", "Needs Improvement", "Poor Form")
        """
        # Extract features
        X = np.array([self.extract_features(kp) for kp in keypoints_list])
        y = np.array([self.class_to_idx[label] for label in labels])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM
        self.svm.fit(X_scaled, y)
        
        # Save model
        self.save_model()
    
    def predict(self, keypoints: np.ndarray) -> Dict[str, any]:
        """
        Predict form classification for given keypoints.
        
        Args:
            keypoints: Shape [1, 17, 3] with [y, x, confidence]
        
        Returns:
            Dictionary with classification, confidence, and probabilities
        """
        features = self.extract_features(keypoints)
        features_scaled = self.scaler.transform([features])
        
        # Predict class
        pred_idx = self.svm.predict(features_scaled)[0]
        pred_class = self.classes[pred_idx]
        
        # Get probabilities
        probs = self.svm.predict_proba(features_scaled)[0]
        prob_dict = {cls: float(prob) for cls, prob in zip(self.classes, probs)}
        
        return {
            "classification": pred_class,
            "confidence": float(probs[pred_idx]),
            "probabilities": prob_dict
        }
    
    def save_model(self):
        """Save the trained model and scaler."""
        with open(self.model_path, 'wb') as f:
            pickle.dump({'svm': self.svm, 'scaler': self.scaler}, f)
    
    def load_model(self):
        """Load a trained model and scaler if available."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.svm = data['svm']
                self.scaler = data['scaler']

def generate_synthetic_training_data(exercise_type: str, num_samples: int = 100) -> tuple:
    """
    Generate synthetic training data for demonstration.
    
    Args:
        exercise_type: Type of exercise
        num_samples: Number of samples per class
    
    Returns:
        Tuple of (keypoints_list, labels)
    """
    keypoints_list = []
    labels = []
    
    for label in ["Good Form", "Needs Improvement", "Poor Form"]:
        for _ in range(num_samples):
            # Create dummy keypoints [1, 17, 3]
            keypoints = np.zeros((1, 17, 3))
            for i in range(17):
                keypoints[0, i, 2] = 0.9  # Confidence
                keypoints[0, i, 0:2] = np.random.normal(0.5, 0.1, 2)  # y, x
            
            # Adjust keypoints based on exercise and form
            if exercise_type == "SQUAT":
                if label == "Good Form":
                    keypoints[0, 13, 0] += 0.2  # Lower left knee
                    keypoints[0, 14, 0] += 0.2  # Lower right knee
                elif label == "Needs Improvement":
                    keypoints[0, 11, 0] -= 0.1  # Higher left hip
                    keypoints[0, 12, 0] -= 0.1  # Higher right hip
                else:  # Poor Form
                    keypoints[0, 5, 0] -= 0.2  # Bend left shoulder
                    keypoints[0, 6, 0] -= 0.2  # Bend right shoulder
            
            elif exercise_type == "DEADLIFT":
                if label == "Good Form":
                    keypoints[0, 11, 0] += 0.1  # Straight left hip
                    keypoints[0, 12, 0] += 0.1  # Straight right hip
                elif label == "Needs Improvement":
                    keypoints[0, 5, 0] -= 0.1  # Slightly bent left shoulder
                    keypoints[0, 6, 0] -= 0.1  # Slightly bent right shoulder
                else:  # Poor Form
                    keypoints[0, 5, 0] -= 0.3  # Heavily bent left shoulder
                    keypoints[0, 6, 0] -= 0.3  # Heavily bent right shoulder
            
            elif exercise_type == "BENCH":
                if label == "Good Form":
                    keypoints[0, 7, 0] += 0.1  # Proper left elbow position
                    keypoints[0, 8, 0] += 0.1  # Proper right elbow position
                elif label == "Needs Improvement":
                    keypoints[0, 9, 1] += 0.1  # Misaligned left wrist
                    keypoints[0, 10, 1] += 0.1  # Misaligned right wrist
                else:  # Poor Form
                    keypoints[0, 7, 0] -= 0.2  # Incorrect left elbow
                    keypoints[0, 8, 0] -= 0.2  # Incorrect right elbow
            
            keypoints_list.append(keypoints)
            labels.append(label)
    
    return keypoints_list, labels

def integrate_form_classifier(pose_detector, keypoints: np.ndarray, exercise_type: str) -> Dict[str, any]:
    """
    Integrate form classifier with pose detector.
    
    Args:
        pose_detector: Instance of PoseDetector
        keypoints: Smoothed keypoints [1, 17, 3]
        exercise_type: Type of exercise
    
    Returns:
        Classification result
    """
    # Initialize classifier if not already done
    if not hasattr(pose_detector, 'form_classifiers'):
        pose_detector.form_classifiers = {}
    
    exercise_type = exercise_type.upper()
    if exercise_type not in pose_detector.form_classifiers:
        pose_detector.form_classifiers[exercise_type] = FormClassifier(exercise_type=exercise_type)
        
        # Train with synthetic data if model doesn't exist
        if not os.path.exists(pose_detector.form_classifiers[exercise_type].model_path):
            keypoints_list, labels = generate_synthetic_training_data(exercise_type)
            pose_detector.form_classifiers[exercise_type].train(keypoints_list, labels)
    
    # Predict form
    return pose_detector.form_classifiers[exercise_type].predict(keypoints)

if __name__ == "__main__":
    # Example usage for testing
    from main import PoseDetector
    from ekf_pose_estimation import integrate_ekf_with_pose_detector
    import cv2
    
    model_path = "singlepose-thunder-tflite-float16.tflite"
    pose_detector = PoseDetector(model_path)
    exercise_type = "SQUAT"
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        square_frame = pose_detector.get_square_frame(frame)
        keypoints_with_scores, _ = pose_detector.detect_pose(square_frame)
        keypoints_with_scores = integrate_ekf_with_pose_detector(pose_detector, keypoints_with_scores)
        
        # Classify form
        result = integrate_form_classifier(pose_detector, keypoints_with_scores, exercise_type)
        
        # Draw results
        frame = pose_detector.draw_keypoints(square_frame, keypoints_with_scores, confidence_threshold=0.2)
        frame = pose_detector.draw_connections(frame, keypoints_with_scores, confidence_threshold=0.2)
        
        # Display classification
        text = f"Form: {result['classification']} ({result['confidence']:.2f})"
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Form Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()