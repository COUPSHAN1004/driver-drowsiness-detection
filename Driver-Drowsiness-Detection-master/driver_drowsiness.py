import cv2
import numpy as np
import dlib
from imutils import face_utils
import winsound  # Windows beep sound
import os  # For cross-platform sound

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize Dlib’s face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status tracking
sleep = 0
drowsy = 0
active = 0
yawn = 0
status = ""
color = (0, 0, 0)

# Function to compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to check if the eye is blinking
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

# Function to compute Mouth Aspect Ratio (MAR) for yawning detection
def mouth_aspect_ratio(mouth):
    A = compute(mouth[2], mouth[10])  # Upper lip to lower lip
    B = compute(mouth[4], mouth[8])   # Upper lip to lower lip
    C = compute(mouth[0], mouth[6])   # Left to right lip corners
    mar = (A + B) / (2.0 * C)
    return mar

# Function to play an alarm sound
def play_alarm():
    try:
        winsound.Beep(1000, 1000)  # Windows beep sound
    except:
        os.system("afplay alarm.wav")  # macOS
        os.system("mpg321 alarm.mp3")  # Linux

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        face_frame = frame.copy()  # Update only if a face is found
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Eye blink detection
        left_blink = blinked(landmarks[36], landmarks[37], 
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], 
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Mouth landmarks for yawning detection (Dlib points 48-68)
        mouth = landmarks[48:68]
        mar = mouth_aspect_ratio(mouth)  # Compute MAR

        # Detect yawning
        if mar > 0.6:  # Adjust threshold if needed
            yawn += 1
            if yawn > 6:
                status = "Yawning !"
                color = (0, 165, 255)
                play_alarm()  # Trigger alarm
        else:
            yawn = 0

        # Drowsiness detection
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                play_alarm()

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                play_alarm()

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        # Display status
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw facial landmarks
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    (cv2.imshow("Drowsiness Detector", frame))
    
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
