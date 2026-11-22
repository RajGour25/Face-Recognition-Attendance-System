import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime


KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

def load_known_faces(directory):
    known_encodings = []
    known_names = []

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Created '{directory}' folder. Please add face images (e.g., Ayush.jpg) and restart.")
        return known_encodings, known_names

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(directory, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
             
                name = os.path.splitext(filename)[0].replace("_", " ").title()
                known_names.append(name)
                print(f"[LOADED] {name}")
            else:
                print(f"[WARNING] No face found in {filename}. Skipping.")

    return known_encodings, known_names


def mark_attendance(name, marked_today):
    if name in marked_today:
        return

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    file_exists = os.path.isfile(ATTENDANCE_FILE)

    with open(ATTENDANCE_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        writer.writerow([name, date_str, time_str])

    marked_today.add(name)
    print(f"[ATTENDANCE MARKED] {name} at {time_str} on {date_str}")


def main():
    print("Initializing Face Recognition Attendance System...")
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)

    if not known_encodings:
        print("[ERROR] No valid faces loaded. Add clear face photos to 'known_faces/' and run again.")
        return

    
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Could not access the webcam.")
        return

    marked_today = set()
    print("[RUNNING] Press 'q' on the camera window to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

   
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    mark_attendance(name, marked_today)

           
            top, right, bottom, left = [coord * 4 for coord in face_location]

            box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Face Recognition Attendance System", frame)

       
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("[STOPPED] Attendance session ended.")


if __name__ == "__main__":
    main()



























