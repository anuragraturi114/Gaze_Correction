import cv2
import dlib
import numpy as np
import imutils

# Load pre-trained face detector and facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Function to calculate angle between three points
def angle_between_points(a, b, c):
    ang = np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        shape = predictor(gray, face)
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

        # Calculate angle between eyes and camera
        left_eye_angle = angle_between_points(landmarks[36], landmarks[37], landmarks[39])
        right_eye_angle = angle_between_points(landmarks[42], landmarks[43], landmarks[45])
        average_angle = (left_eye_angle + right_eye_angle) / 2

        # Adjust eyes to look at the camera (you need to implement this part)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
