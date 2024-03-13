import cv2
import dlib
import numpy as np

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye.xml")


def correct_gaze(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("Error: No face detected")
        return None

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)

        # Check if eye landmarks are valid
        if left_eye is None or right_eye is None:
            print("Error: Unable to detect eye landmarks")
            return None

        # Calculate the angle between the eyes
        dy = right_eye.y - left_eye.y
        dx = right_eye.x - left_eye.x
        angle = np.arctan2(dy, dx) * 180.0 / np.pi

        # Rotate the image to correct the gaze
        M = cv2.getRotationMatrix2D((left_eye.x, left_eye.y), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

        return rotated_image


if __name__ == "__main__":
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is empty
        if not ret:
            print("Error: Unable to capture frame")
            continue

        # Correct the gaze for the current frame
        corrected_frame = correct_gaze(frame)

        # Check if the corrected frame is valid
        if corrected_frame is not None:
            # Display the corrected frame
            cv2.imshow("Gaze Correction", corrected_frame)
        else:
            print("Error: Unable to correct gaze")

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
