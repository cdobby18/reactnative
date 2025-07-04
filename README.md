# TECH TITANS - POWERLIFT
 
**Powerlift: Enhancing Pose Estimation Accuracy in Powerlifting Performance Analysis**

## Overview

**Powerlift** is a research-based mobile and backend system that leverages computer vision to analyze and enhance powerlifting performance. By combining pose estimation, barbell detection, and advanced filtering/classification techniques, the system provides real-time feedback and form evaluation for lifters, aiming to minimize injury risks and optimize training efficiency.

---

## Key Features

- **Real-Time Pose Estimation** using **MoveNet** for tracking joint positions during lifts.
- **Barbell Detection** using **YOLOv8** to track the barbell's movement throughout the lift.
- **Pose Smoothing** with **Extended Kalman Filter** to reduce noise in pose keypoints and improve temporal accuracy.
- **Lifting Form Classification** using **Multiclass Support Vector Machine (SVM)** to assess technique quality (Good, Needs Improvement, etc.)
- **Instant Feedback** of performance metrics.

---

## Algorithms Used

| Component              | Algorithm/Model           |
|------------------------|---------------------------|
| Pose Estimation        | MoveNet                   |
| Object Detection       | YOLOv8                    |
| Temporal Filtering     | Extended Kalman Filter    |
| Form Classification    | Multiclass SVM            |

---

## Tech Stack

- **Backend:** Python (FastAPI), TensorFlow, PyTorch, OpenCV
- **Frontend:** React Native (Expo)
- **ML/DL Models:** MoveNet, YOLOv8, Scikit-learn
- **Deployment:** Localhost/Mobile Testing Environment

---

## Team Members

- **Project Manager:** Edilson Tuban  
- **Full Stack Developers:** Carl Joshua Coloma, Rey Justine Morales  
- **Frontend Developer & Database Administrator:** Jan Jayson Yap

---

## Exercises

- Squat  
- Bench Press  
- Deadlift  

Each lift is recorded and analyzed for:
- Bar path tracking
- Joint angle accuracy
- Range of motion
- Classification of form

