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



def call_repeatedly(interval, func, *args):
    def loop():
        while not stop_event.is_set():
            func(*args)  # Pass the scaler and classifier to the function
            stop_event.wait(interval)
    Thread(target=loop).start()


prediction = 'None'
image = []
tello = tellopy.Tello()
drone_is_flying = False

def control():
    global prediction, tello, drone_is_flying
    # drone_is_flying = True
    if prediction:
        if prediction in ["Palm", "Fist","Rock","Ok","Peace","Like", "Up", "Down", "None"]:
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


def recognise_gesture():
    global prediction, image
    print("it is activated every second")
    gestures = {
        0: "Palm",#dlan
        1: "Fist",#pst
        2: "Rock",#rock
        3: "Ok",#ok
        4: "Peace",#peace
        5: "Like",#palec hore
        6: "Up",#ukazovak hore
        7: "Down",#ukazovak dole
        8: "None"
        }
    # prediction = 'None'
    mp_hands = mp.solutions.hands
    #prediction = 'None'
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
        control()
        print(prediction)


def main():
    global prediction, image, tello, drone_is_flying
    gesture_recognition_started = False
    drone_is_flying = False

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
        frame_skip = 300

        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                key = cv2.waitKey(1)
                image = np.array(frame.to_image())

                tello.palm_land()

                if key == ord("g"):
                    call_repeatedly(5, recognise_gesture)
                # if key == ord("s"):
                #     control()

                if key == ord('q'):

                    break
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(image, prediction,(10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
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