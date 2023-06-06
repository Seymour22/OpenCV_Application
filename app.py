import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from tensorflow import keras
import tempfile


actions = np.array(['Left', 'Right'])
colors = [(245, 117, 16), (117, 245, 16)]

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities





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
    st.title("Welcome! This application can detect if a face turns left or right.")
    st.text('Please turn your head left or right for me to detect.')



    model = keras.models.load_model('action.h5')

    # Create sidebar with options for user to choose
    option = st.selectbox("Choose an option", ("Webcam", "Upload video"))
    fps = 0
    i = 0
    stframe = st.empty()

    # Depending on user's option, show webcam or file uploader
    if option == "Webcam":
        sequence = []
        


    else:

        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
        if video_file is not None:
            # Save the uploaded video file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(video_file.read())

            # Read the video file using OpenCV
            vid = cv2.VideoCapture(temp_path)
            # Set mediapipe model
            sequence = []
            sentence = []
            predictions = []
            threshold = 0.5



            with mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as holistic_mesh:

                # Set mediapipe model

                while vid.isOpened():
                    ret, frame = vid.read()
                    if not ret:
                        break

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
                        frame = prob_viz(res, actions, frame, colors)

                    cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(frame, ' '.join(sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    stframe.image(frame, channels='BGR', use_column_width=True)


if __name__ == "__main__":
    main()



