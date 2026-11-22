# Face Recognition Attendance System

A real-time Python-based attendance tracking system built using **OpenCV** and the **face_recognition** library. It detects faces via webcam, matches them against stored images, and automatically records attendance with date and timestamp in a CSV file.

## Features
- Real-time face detection and recognition using webcam.
- Automatic folder creation (`known_faces/`) if not present.
- Prevents duplicate attendance entries within the same session.
- Logs Name, Date, and Time directly into `attendance.csv`.

## Tech Stack
- **Language:** Python 3
- **Libraries:** `opencv-python`, `face-recognition`, `numpy`, `csv`

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Face-Recognition-Attendance-System.git](https://github.com/your-username/Face-Recognition-Attendance-System.git)
