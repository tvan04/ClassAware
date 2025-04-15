import threading
import cv2
import csv
import time
from deepface import DeepFace
import mediapipe as mp

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference_img = cv2.imread('image.jpg')

# Initialize CSV log file
log_file = "participation_log.csv"
with open(log_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Event"])

def log_event(event):
    """Logs an event with a timestamp."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event])

def check_face(frame):
    """Checks if the face matches the reference image."""
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
            log_event("No match detected")
    except ValueError:
        face_match = False
        log_event("No match detected (error)")

# Gesture state flags
thumbs_up_logged = False
thumbs_down_logged = False
raised_hand_logged = False
gesture_text = ""  # Text to display current gesture

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face
        face_results = face_detection.process(rgb_frame)
        face_y = None

        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                face_y = bbox.ymin * frame.shape[0]  # Convert to absolute Y position

        # Detect hands
        results = hands.process(rgb_frame)

        # Reset gesture text every frame; will be set if gesture detected
        gesture_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                wrist_y = wrist.y * frame.shape[0]  # Convert to absolute position

                # Raised Hand Detection (wrist above face)
                if face_y is not None and wrist_y < face_y:
                    if not raised_hand_logged:
                        log_event("Raised Hand detected")
                        raised_hand_logged = True
                    gesture_text = "HAND RAISED"
                else:
                    raised_hand_logged = False  # Reset when no longer detected

                # Thumbs Up Detection
                if thumb_tip.y < index_tip.y and thumb_tip.y < wrist.y:
                    if not thumbs_up_logged:
                        log_event("Thumbs Up detected")
                        thumbs_up_logged = True
                    thumbs_down_logged = False  # Reset other gesture
                    gesture_text = "THUMBS UP"
                else:
                    thumbs_up_logged = False  # Reset when no longer detected

                # Thumbs Down Detection
                if thumb_tip.y > index_tip.y and thumb_tip.y > wrist.y:
                    if not thumbs_down_logged:
                        log_event("Thumbs Down detected")
                        thumbs_down_logged = True
                    thumbs_up_logged = False  # Reset other gesture
                    gesture_text = "THUMBS DOWN"
                else:
                    thumbs_down_logged = False  # Reset when no longer detected

        # Perform face verification every 30 frames
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        # Display gesture text (if any)
        if gesture_text:
            cv2.putText(frame, gesture_text, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 0), 3)

        # Display match status
        status_text = "MATCH!" if face_match else "NO MATCH!"
        status_color = (0, 255, 0) if face_match else (0, 0, 255)
        cv2.putText(frame, status_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 3)

        cv2.imshow('Video', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()