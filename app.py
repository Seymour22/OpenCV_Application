import cv2
import numpy as np
import os
import time
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import streamlit as st

from tensorflow import keras



actions = np.array(['Left', 'Right'])

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

mp_face_mesh = mp.solutions.face_mesh

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5
        prevTime = 0

        i =self.i+1
        model = keras.models.load_model('action.h5')
        with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic_mesh:

            # Set mediapipe model

                #ret, frame = vid.read()
                # if not ret:
                #     break

                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # results = holistic_mesh.process(frame)
                #
                # frame.flags.writeable = True
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame, results = mediapipe_detection(frame, holistic_mesh)

                draw_landmarks(frame, results)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-3:]

                if len(sequence) == 3:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))

                    # 3. Viz logic
                    if np.unique(predictions[-1:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:

                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(res, actions, frame, colors)

                cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # mp_drawing.draw_landmarks(
                #     image=frame,
                #     landmark_list=results.face_landmarks,
                #     connections=mp_holistic.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=drawing_spec,
                #     connection_drawing_spec=drawing_spec)
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                #stframe.image(frame,channels = 'BGR',use_column_width=True)
            # vid.release()
            # cv2.destroyAllWindows()

        return frame





def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def main():
    actions = np.array(['Left', 'Right'])

    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    mp_face_mesh = mp.solutions.face_mesh
    st.title("Welcome! This application can detect if a face turns left or right.")
    st.text('Please turn your head left or right for me to detect.')
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})





    # Create sidebar with options for user to choose
    option = st.selectbox("Choose an option", ("Webcam", "Upload video"))
    fps = 0
    i = 0
    stframe = st.empty()

    # Depending on user's option, show webcam or file uploader
    if option == "Webcam":

        camera_options = []

        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
        #webrtc_ctx = webrtc_streamer(key="example", rtc_configuration=rtc_configuration, mode=WebRtcMode.SENDRECV,
                                     #video_processor_factory=VideoTransformerBase)



          # open cv2 capture object
        #vid = cv2.VideoCapture(i)
        # if vid.isOpened():
        #     camera_options.append(f"Camera {i}")
        # #vid.release()
        # prevTime = 0
        # option = st.selectbox("Choose a camera", camera_options)

        #vid = cv2.VideoCapture(int(option.split(" ")[-1]))
        #vid = cv2.VideoCapture(1)


        # webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION)
        # vid = webrtc_ctx




    else:


        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
        if video_file is not None:
            # Read the video file using OpenCV
            cap = cv2.VideoCapture(video_file)
            # Set mediapipe model
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

                while cap.isOpened():

                    # Read feed
                    ret, frame = cap.read()

                    # Check if end of video
                    if not ret:
                        break

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_landmarks(image, results)

                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)



if __name__ == "__main__":
    main()



