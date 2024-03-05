import cv2
import mediapipe as mp
import time
from model import KeyPointClassifier
import landmark_utils as u

# Constants for model configuration and gesture recognition interval
MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
GESTURE_RECOGNITION_INTERVAL = 5  # Seconds

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands.Hands(
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=MIN_DETECTION_CONFIDENCE)

class GestureRecognizer:
    def __init__(self):
        self.prediction = 'None'
        self.kpclf = KeyPointClassifier()
        self.last_gesture_time = time.time() - GESTURE_RECOGNITION_INTERVAL  # Allow immediate recognition
        self.gesture_recognized_previously = False

    def recognise_gesture(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = mp_hands.process(image_rgb)
        gesture_recognized = False

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmark_list = u.calc_landmark_list(image_rgb, hand_landmarks)
                keypoints = u.pre_process_landmark(landmark_list)
                gesture_index = self.kpclf(keypoints)
                self.prediction = self.get_gesture_name(gesture_index)
                gesture_recognized = True
                print(self.prediction)
                break

        if not gesture_recognized:
            self.prediction = 'None'

        if not gesture_recognized and self.gesture_recognized_previously:
            self.last_gesture_time = time.time()
            self.gesture_recognized_previously = False
        elif gesture_recognized:
            self.gesture_recognized_previously = True

    @staticmethod
    def get_gesture_name(index):
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
        return gestures.get(index, "None")

def find_face(image):
    face_results = mp_face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ih, iw, _ = image.shape
    face_area_percentage = 0
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            face_width = bboxC.width * iw
            face_height = bboxC.height * ih
            face_area = face_width * face_height
            frame_area = iw * ih
            face_area_percentage = (face_area / frame_area) * 100
            print(f"Face area: {face_area} pixels, which is {face_area_percentage:.2f}% of the frame")
            break
    return face_area_percentage


def is_hand_near_face(hand_results, face_results, image_width, image_height, threshold=0.75):
    """
    Checks if any hand is near any detected face based on the bounding boxes.
    :param hand_results: MediaPipe Hands detection results.
    :param face_results: MediaPipe Face detection results.
    :param image_width: The width of the image being processed.
    :param image_height: The height of the image being processed.
    :param threshold: Proximity threshold as a fraction of image width/height.
    :return: True if any hand is near a face, False otherwise.
    """
    if not hand_results.multi_hand_landmarks or not face_results.detections:
        return False  # Early exit if no hands or no faces detected

    for face_detection in face_results.detections:
        bboxC = face_detection.location_data.relative_bounding_box
        face_xmin = bboxC.xmin * image_width
        face_ymin = bboxC.ymin * image_height
        face_width = bboxC.width * image_width
        face_height = bboxC.height * image_height

#center ruky
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                hand_x = landmark.x * image_width
                hand_y = landmark.y * image_height

                # Simple proximity check based on bounding box and threshold
                if (face_xmin - face_width * threshold) < hand_x < (face_xmin + face_width + face_width * threshold) and \
                        (face_ymin - face_height * threshold) < hand_y < (
                        face_ymin + face_height + face_height * threshold):
                    return True
        return False


def main():
    cap = cv2.VideoCapture(0)
    gesture_recognizer = GestureRecognizer()

    recognition_paused = False  # New flag to track if recognition is paused


    while True:
        ret, image = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = mp_hands.process(image_rgb)
        face_results = mp_face_detection.process(image_rgb)

        frame_height, frame_width, _ = image.shape
        hand_near_face = False

        hand_near_face = is_hand_near_face(hand_results, face_results, frame_width, frame_height)

        if hand_near_face:
            gesture_recognizer.recognise_gesture(image)
            last_time_gesture_recognized = time.time()  # Update the time when a gesture is recognized
        else:
            gesture_recognizer.prediction = 'None'

        now = time.time()
        if now - last_time_gesture_recognized > NO_GESTURE_TIMEOUT:
            # Extract face position and area information
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    face_x = (bboxC.xmin + bboxC.width / 2) * frame_width
                    face_y = (bboxC.ymin + bboxC.height / 2) * frame_height
                    face_area = bboxC.width * bboxC.height  # Relative area

                    follow_face(face_x, face_y, face_area, frame_width, frame_height)
                    break

    face_area_percentage = find_face(image)

        now = time.time()
        elapsed_time_since_last_recognition = now - gesture_recognizer.last_gesture_time
        elapsed_time_text = f"Time since last recognition: {elapsed_time_since_last_recognition:.2f}s"

        display_text = "Recognition paused. Hand too far from face." if not hand_near_face else gesture_recognizer.prediction
        cv2.putText(image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        face_area_text = f"Face Area: {face_area_percentage:.2f}%"
        cv2.putText(image, face_area_text, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (204, 0, 102), 2)
        cv2.putText(image, elapsed_time_text, (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (204, 0, 102), 2)

        cv2.imshow('Gesture Recognition', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()