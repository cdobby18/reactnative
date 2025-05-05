import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.connections = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7),
            (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13),
            (12, 14), (13, 15), (14, 16), (11, 12)
        ]

    def preprocess_frame(self, frame):
        """Preprocess the frame for pose detection"""
        img = cv2.resize(frame, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def detect_pose(self, frame):
        """Detect pose keypoints in the frame"""
        input_data = self.preprocess_frame(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        return keypoints_with_scores, frame

    def draw_keypoints(self, frame, keypoints_with_scores, confidence_threshold=0.2):
        """Draw keypoints on the frame"""
        height, width, _ = frame.shape
        shaped = np.squeeze(keypoints_with_scores)

        for i, (y, x, confidence) in enumerate(shaped):
            if confidence > confidence_threshold:
                x_pixel = int(x * width)
                y_pixel = int(y * height)
                cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
                cv2.putText(frame, self.keypoint_names[i], (x_pixel + 10, y_pixel),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame

    def draw_connections(self, frame, keypoints_with_scores, confidence_threshold=0.2):
        """Draw connections between keypoints on the frame"""
        height, width, _ = frame.shape
        shaped = np.squeeze(keypoints_with_scores)

        for connection in self.connections:
            idx1, idx2 = connection
            y1, x1, confidence1 = shaped[idx1]
            y2, x2, confidence2 = shaped[idx2]

            if confidence1 > confidence_threshold and confidence2 > confidence_threshold:
                x1_pixel = int(x1 * width)
                y1_pixel = int(y1 * height)
                x2_pixel = int(x2 * width)
                y2_pixel = int(y2 * height)
                cv2.line(frame, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 255, 0), 2)
        return frame

    def analyze_exercise(self, keypoints_with_scores, exercise):
        """Analyze the exercise based on keypoints"""
        feedback = ""
        shaped = np.squeeze(keypoints_with_scores)

        if exercise == "SQUAT":
            left_hip = shaped[self.keypoint_names.index('left_hip')]
            left_knee = shaped[self.keypoint_names.index('left_knee')]
            left_ankle = shaped[self.keypoint_names.index('left_ankle')]

            # Simple angle calculation for knee
            angle = self.calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
            if angle > 90:
                feedback = "Squat deeper for better form!"
            else:
                feedback = "Good squat depth!"

        elif exercise == "DEADLIFT":
            left_hip = shaped[self.keypoint_names.index('left_hip')]
            left_knee = shaped[self.keypoint_names.index('left_knee')]
            left_shoulder = shaped[self.keypoint_names.index('left_shoulder')]

            # Check back angle for deadlift
            back_angle = self.calculate_angle(left_shoulder[:2], left_hip[:2], left_knee[:2])
            if back_angle < 45:
                feedback = "Keep your back straighter during the deadlift!"
            else:
                feedback = "Good back position!"

        return {"feedback": feedback}

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    def analyze_with_barbell(self, keypoints_with_scores, barbell_results, exercise):
        """Analyze exercise with barbell information"""
        feedback = self.analyze_exercise(keypoints_with_scores, exercise).get("feedback", "")
        
        if barbell_results and barbell_results.get('boxes'):
            barbell_detected = True
            barbell_box = barbell_results['boxes'][0] if barbell_results['boxes'] else None
            if barbell_box:
                barbell_center_y = (barbell_box[1] + barbell_box[3]) / 2
                
                shaped = np.squeeze(keypoints_with_scores)
                left_hip = shaped[self.keypoint_names.index('left_hip')]
                left_hip_y = left_hip[0]

                if exercise == "DEADLIFT" and barbell_center_y > left_hip_y:
                    feedback += " Keep the barbell closer to your body!"
        
        return {"feedback": feedback}

class BarbellDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Detect barbell in the frame using YOLO"""
        results = self.model(frame)
        return {
            "boxes": [[float(box.xyxy[0][0]), float(box.xyxy[0][1]), 
                       float(box.xyxy[0][2]), float(box.xyxy[0][3])] 
                      for box in results[0].boxes],
            "keypoints": None  # YOLOv8 might not return keypoints directly
        }

    def draw_on_frame(self, frame, results):
        """Draw barbell detections on the frame"""
        for box in results.get('boxes', []):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Barbell", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

def get_square_frame(frame):
    """Convert frame to square by padding"""
    height, width, _ = frame.shape
    size = max(height, width)
    square_frame = np.zeros((size, size, 3), dtype=np.uint8)
    
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    
    square_frame[y_offset:y_offset + height, x_offset:x_offset + width] = frame
    return square_frame