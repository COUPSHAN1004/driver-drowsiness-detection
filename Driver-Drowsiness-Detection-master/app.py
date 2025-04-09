from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import dlib
from imutils import face_utils
import winsound  # Windows beep sound
import os  # For cross-platform sound
import threading  # For handling start/stop


app = Flask(__name__)

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Global variables for status tracking
sleep = 0
drowsy = 0
active = 0
yawn = 0
status = "Initializing..."
color = (0, 0, 0)

# Flag to control detection start/stop
detection_running = False

# Function to compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to check if the eye is blinking
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Eyes open
    elif ratio > 0.21 and ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Sleeping

# Function to compute Mouth Aspect Ratio (MAR) for yawning detection
def mouth_aspect_ratio(mouth):
    A = compute(mouth[2], mouth[10])  # Upper lip to lower lip
    B = compute(mouth[4], mouth[8])   # Upper lip to lower lip
    C = compute(mouth[0], mouth[6])   # Left to right lip corners
    mar = (A + B) / (2.0 * C)
    return mar

# Function to play alarm sound
def play_alarm():
    try:
        winsound.Beep(1000, 1000)  # Windows beep sound
    except:
        os.system("afplay alarm.wav")  # macOS
        os.system("mpg321 alarm.mp3")  # Linux

# Initialize camera
cap = cv2.VideoCapture(0)

# Function to generate frames for video streaming
def generate_frames():
    global sleep, drowsy, active, yawn, status, color, detection_running
    while True:
        success, frame = cap.read()
        if not success:
            break

        if detection_running:  # Only run detection when enabled
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                # Eye blink detection
                left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
                right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

                # Mouth aspect ratio for yawning detection
                mouth = landmarks[48:68]
                mar = mouth_aspect_ratio(mouth)

                # Detect yawning
                if mar > 0.6:
                    yawn += 1
                    if yawn > 6:
                        status = "Yawning!"
                        color = (0, 165, 255)
                        play_alarm()
                else:
                    yawn = 0

                # Drowsiness detection
                if left_blink == 0 or right_blink == 0:
                    sleep += 1
                    drowsy = 0
                    active = 0
                    if sleep > 6:
                        status = "SLEEPING!!!"
                        color = (255, 0, 0)
                        play_alarm()

                elif left_blink == 1 or right_blink == 1:
                    sleep = 0
                    active = 0
                    drowsy += 1
                    if drowsy > 6:
                        status = "Drowsy!"
                        color = (0, 0, 255)
                        play_alarm()

                else:
                    drowsy = 0
                    sleep = 0
                    active += 1
                    if active > 6:
                        status = "Active :)"
                        color = (0, 255, 0)

                # Display status on frame
                cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to start detection
@app.route('/start')
def start_detection():
    global detection_running, status
    detection_running = True
    status = "Detection Running..."
    return jsonify({"message": "Detection Started"})

# Route to stop detection
@app.route('/stop')
def stop_detection():
    global detection_running, status
    detection_running = False
    status = "Detection Stopped"
    return jsonify({"message": "Detection Stopped"})

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
