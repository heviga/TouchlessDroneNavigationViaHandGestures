import cv2
import mediapipe as mp
from model import KeyPointClassifier
import landmark_utils as u
import sys
import traceback
import tellopy
from threading import Thread, Event
import time
import numpy as np
import av
stop_event = Event()

image = []
tello = tellopy.Tello()
drone_is_flying = False
face_detected = False
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5)
pError = 0
last_recognition_time = 0
recognition_interval = 5.0
last_face_detection_time = 0  # Timestamp of the last face detection


def call_repeatedly(interval, func, *args):
    def loop():
        while not stop_event.is_set():
            func(*args)  # Pass the scaler and classifier to the function
            stop_event.wait(interval)
    Thread(target=loop).start()


# defining commands for drone
def control():
    global prediction, tello, drone_is_flying
    # drone_is_flying = True
    if prediction:
        if prediction in ["Palm", "Fist", "Rock", "Ok", "Peace", "Like", "Up", "Down", "None"]:
            if prediction == 'Fist':
                tello.forward(30)
                time.sleep(1)
                tello.forward(0)
            elif prediction == 'Palm':
                tello.backward(30)
                time.sleep(1)
                tello.backward(0)
            elif prediction == 'Rock':
                tello.flip_forward()
            elif prediction == 'Peace':
                cv2.imwrite("picture.png", image)
            elif prediction == "Ok":
                if drone_is_flying:
                    tello.land()
                    drone_is_flying = False
                else:
                    tello.takeoff()
                    drone_is_flying = True
            elif prediction == 'Like':
                tello.clockwise(180)
                time.sleep(2)
                tello.clockwise(180)
                time.sleep(2)
                tello.clockwise(0)
            elif prediction == 'Up':
                tello.up(30)
                time.sleep(1)
                tello.up(0)
            elif prediction == 'Down':
                tello.down(30)
                time.sleep(1)
                tello.down(0)
        else:
            pass
    else:
        prediction = 'No Hand Detected'
        pass

    return prediction

# if it recognizes gesture, drone executes command


def recognise_gesture():
    global prediction, image, face_detected, last_recognition_time
    current_time = time.time()
    print(f"it is activated every {recognition_interval} second(s)")

    if current_time - last_recognition_time < recognition_interval:
        return  # not enough time has passed, skip recognition

    gestures = {
        0: "Palm",  # dlan
        1: "Fist",  # pst
        2: "Rock",  # rock
        3: "Ok",  # ok
        4: "Peace",  # peace
        5: "Like",  # palec hore
        6: "Up",  # ukazovak hore
        7: "Down",  # ukazovak dole
        8: "None"
    }
    # prediction = 'None'
    mp_hands = mp.solutions.hands
    # prediction = 'None'
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    kpclf = KeyPointClassifier()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        print("it finded landmarks!")
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = u.calc_landmark_list(image, hand_landmarks)
            keypoints = u.pre_process_landmark(landmark_list)
            gesture_index = kpclf(keypoints)

            prediction = gestures[gesture_index]
            last_recognition_time = current_time
        control()
        print(prediction)


def searching():
    global face_results, tello
    if not face_results.detections:
        print("Starting search...")
        time.sleep(2)
        tello.clockwise(50)
        time.sleep(1)
        tello.clockwise(0)
        print("Search stopped, re-checking for faces...")


def find_face():
    global image, tello, face_results, pError, prediction, face_detected, last_face_detection_time
    current_time = time.time()

    face_results = mp_face_detection.process(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ih, iw, _ = image.shape
    face_area_percentage = 0

    if face_detected:
        print("Face detected, stopping search.")
        recognise_gesture()
        return

    if face_results.detections:
        face_detected = True
        last_face_detection_time = current_time
        tello.clockwise(0)  # Stop rotating
        time.sleep(5)

        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x0, y0 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            face_width = int(bboxC.width * iw)
            face_height = int(bboxC.height * ih)
            # cv2.rectangle(image, (x0, y0), (x0 + face_width, y0 + face_height), (0, 255, 0), 2)
            face_area = face_width * face_height
            frame_area = iw * ih
            face_area_percentage = (face_area / frame_area) * 100
            print(
                f"Face area: {face_area} pixels, which is {face_area_percentage:.2f}% of the frame")

            break
    else:   # If no faces are detected for more than 10s
        if current_time - last_face_detection_time > 10:
            if face_detected:  # Check is optional, but avoids unnecessary prints
                print("No face detected for 10 seconds, resetting...")
                face_detected = False
            searching()
    return face_area_percentage

# def check_for_face_timeout():
#     global last_face_detection_time, face_detected
#     current_time = time.time()
#     if face_detected and (current_time - last_face_detection_time > 10):
#         print("No face detected for 10 seconds, resetting...")
#         face_detected = False


def main():
    global image, tello, drone_is_flying, prediction
    drone_is_flying = False
    face_detected = False
    # Initialize the time when the last gesture was recognized

    try:
        tello.connect()
        tello.wait_for_connection(60.0)
        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(tello.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        tello.takeoff()
        drone_is_flying = True

        # vyletime trochu vysssie jedenkrat
        time.sleep(3)
        tello.up(30)
        time.sleep(2)
        tello.up(0)
        print("tello went up")
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                key = cv2.waitKey(1)
                image = np.array(frame.to_image())
                # tello.palm_land()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # hladanie
                if key == ord("f"):
                    call_repeatedly(1, find_face)
                    # check_for_face_timeout()

                if key == ord('q'):
                    break
                    tello.land()

                # cv2.putText(image, prediction,(10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
                cv2.imshow('MediaPipe Hands', image)

            if key == ord('q'):
                break
        tello.land()
        drone_is_flying = False

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)

    finally:
        tello.land()
        tello.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
