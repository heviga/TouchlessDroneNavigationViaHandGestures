

#POKUS 3
import cv2
import mediapipe as mp
import numpy as np
import sys
import traceback
from threading import Thread, Event
import time

from model import KeyPointClassifier
import landmark_utils as u

stop_event = Event()

# Initialize MediaPipe solutions for hands and face detection
mp_hands = mp.solutions.hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

prediction = 'None'
last_gesture_time = time.time()

def call_repeatedly(interval, func, *args):
    def loop():
        while not stop_event.is_set():
            func(*args)
            stop_event.wait(interval)
    Thread(target=loop).start()



def recognise_gesture(image):
    global prediction, mp_hands, last_gesture_time
    gestures = {
        0: "Palm",
        1: "Fist",
        2: "Rock",
        3: "Ok",
        4: "Peace",
        5: "Like",
        6: "Up",
        7: "Down",
        8: "None"
    }

    kpclf = KeyPointClassifier()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format

    hand_results = mp_hands.process(image_rgb)  # Process the RGB image
    hands = []


    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmark_list = u.calc_landmark_list(image_rgb, hand_landmarks)
            keypoints = u.pre_process_landmark(landmark_list)
            gesture_index = kpclf(keypoints)
            prediction = gestures[gesture_index]
            print(prediction)
            break  # Process first detected hand for simplicity
            palm_base = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
            hands.append(palm_base)
    if prediction is not None:
        last_gesture_time = time.time()

    return hands


def find_face(image, mp_face_detection):
    face_area_percentage = 0  # Define at the start to ensure it's always defined
    face_results = mp_face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    faces = []
    ih, iw, _ = image.shape  # Frame dimensions
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            face_width = bboxC.width * iw
            face_height = bboxC.height * ih
            face_area = face_width * face_height
            frame_area = iw * ih
            face_area_percentage = (face_area / frame_area) * 100
            print(f"Face area: {face_area} pixels, which is {face_area_percentage:.2f}% of the frame")
            faces.append(bboxC)
            break  # Assuming only the first detected face is of interest
    return faces, face_area_percentage




def is_hand_next_to_face(hands, faces, image_width, image_height):
    for hand in hands:
        hx = hand.x * image_width
        hy = hand.y * image_height
        for face in faces:
            fx = face.xmin * image_width
            fy = face.ymin * image_height
            fw = face.width * image_width
            fh = face.height * image_height
            # Simple check: if the hand is within the face bounding box expanded by some margin
            margin = 0.1  # Adjust based on your needs
            if fx - margin * fw < hx < fx + (1 + margin) * fw and fy - margin * fh < hy < fy + (1 + margin) * fh:
                return True

    return False

hand_near_face = False
def check_hand_face_proximity(image):
    global hand_near_face
    hands = recognise_gesture(image)
    faces = find_face(image, mp_face_detection)
    frame_height, frame_width, _ = image.shape
    if is_hand_next_to_face(hands, faces, frame_width, frame_height):
        print("Hand is next to a face.")
        hand_near_face = True
    else:
        print("Hand not near face")
        hand_near_face = False


def main():
    global prediction, last_gesture_time, hand_near_face, face_area_percentage
    cap = cv2.VideoCapture(0)

    hand_near_face_start = None  # Track when the hand first gets near the face

    last_gesture_time = time.time() - 5  # Allow immediate recognition


    while True:
        ret, image = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Perform the proximity check with the current frame
        hand_near_face = check_hand_face_proximity(image)

        _, face_area_percentage = find_face(image, mp_face_detection)  # Updated to correctly capture the variable

        now = time.time()

        if hand_near_face and (now - last_gesture_time >=5):
                recognise_gesture(image)
                last_gesture_time = now

        # Calculate the time elapsed since the last gesture recognition
        elapsed_time_since_last_recognition = now - last_gesture_time
        elapsed_time_text = f"Time since last recognition: {elapsed_time_since_last_recognition:.2f}s"


        cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (204, 0, 102), 2)
        face_area_text = f"Face Area: {face_area_percentage:.2f}%"
        cv2.putText(image, face_area_text, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (204, 0, 102), 2)
        cv2.putText(image, elapsed_time_text, (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (204, 0, 102), 2)


        cv2.imshow('MediaPipe Hands and Face Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
