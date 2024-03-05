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


image = []
tello = tellopy.Tello()
drone_is_flying = False
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
pError = 0

def searching():
    global face_results, tello
    if not face_results.detections:
        print("Starting search...")
        time.sleep(2)
        tello.clockwise(50)
        time.sleep(1)
        tello.clockwise(0)
        time.sleep(1)
        print("Search stopped, re-checking for faces...")


def find_face():
    global image, tello, face_results, pError
    face_results = mp_face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ih, iw, _ = image.shape
    face_area_percentage = 0

    if face_results.detections:
        print("Face detected, stopping search.")
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x0,y0 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            face_width = int(bboxC.width * iw) #
            face_height = int(bboxC.height * ih)
            # cv2.rectangle(image, (x0, y0), (x0 + face_width, y0 + face_height), (0, 255, 0), 2)
            face_area = face_width * face_height
            frame_area = iw * ih
            face_area_percentage = (face_area / frame_area) * 100
            print(f"Face area: {face_area} pixels, which is {face_area_percentage:.2f}% of the frame")

            break
    else:
        # If no faces are detected
        print("No, not a face, keep searching")
        searching()
    return face_area_percentage


def main():
    global  image, tello, drone_is_flying
    drone_is_flying = False
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
        time.sleep(3)
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



                #hladanie
                if key == ord("f"):
                    call_repeatedly(5, find_face)

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