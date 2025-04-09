Driver Drowsiness Detection System
A real-time Driver Drowsiness and Yawning Detection System built using Flask, OpenCV, Dlib, and Facial Landmarks. The system uses a webcam to monitor the driver's eyes and mouth, detect signs of fatigue or sleep, and trigger an audio alert if needed.

Features
Real-time face and eye detection using dlib and OpenCV
Detects:
Eye closure (sleep)
Partial eye closure (drowsiness)
Yawning (via mouth aspect ratio)
Plays an audio alert when drowsiness or yawning is detected
Web interface to start/stop detection
Live webcam video stream in the browser
Project Structure
DRIVER-DROWSINESS-DETECTION/ │ ├── static/ │ └── alarm.mp3 # Alarm sound file │ ├── templates/ │ └── index.html # Web interface │ ├── app.py # Main Flask app with detection ├── driver_drowsiness.py # Optional: Modular detection logic ├── shape_predictor_68_face_landmarks.dat # Dlib landmark model └── README.md

Requirements
Python 3.7+
OpenCV
Flask
dlib
imutils
numpy
Install with:

pip install opencv-python flask dlib imutils numpy


Setup Instructions
Clone the repository:
git clone https://github.com/COUPSHAN1004/driver-drowsiness-detection.git
cd driver-drowsiness-detection

Download the shape predictor model:

Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Extract and place shape_predictor_68_face_landmarks.dat in the project directory.

Run the app:
python app.py

Open your browser and go to:
http://127.0.0.1:5000/
Click “Start Detection” to begin monitoring.

 Detection Logic
Eye Aspect Ratio (EAR):

0.25: Eyes Open

0.21–0.25: Drowsy

< 0.21: Sleeping

Mouth Aspect Ratio (MAR):

0.6 for several frames: Yawning

Audio alerts are triggered for drowsiness, sleep, or yawning.


 Future Improvements
Add phone usage detection (using MobileNet SSD or YOLO)

Add driver distraction and head tilt detection

Deploy as a desktop app or on Raspberry Pi

Create a dashboard with logs and analysis


License
This project is licensed under the MIT License. Feel free to use and modify.

🙌 Author
Made with ❤️ By Swasti Sawarnik and Vanshika Goyal# driver-drowsiness-detection
