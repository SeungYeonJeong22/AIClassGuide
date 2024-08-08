import cv2
import asyncio
import websockets
import pickle
from deepface import DeepFace
import streamlit as st
import numpy as np

# 서버 주소 설정
WEBSOCKET_HOST = '192.168.0.12'
WEBSOCKET_PORT = 8502
# STREAMLIT_HOST = '192.168.0.12'
# STREAMLIT_PORT = 8502

def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.6  # Adjust the transparency of the overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # White rectangle
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 15  # Where the first text is put into the overlay
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_position += 20

    return frame

async def send_video():
    uri = f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}"
    try:
        async with websockets.connect(uri) as websocket:
            cap = cv2.VideoCapture(0)

            st.title("Student's Live Stream with Emotion Analysis")
            stframe = st.empty()

            # Define the stop checkbox outside the loop
            stop = st.checkbox('Stop', key='stop_checkbox')

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to reduce size
                frame = cv2.resize(frame, (320, 240))

                # Analyze the frame using DeepFace
                try:
                    # result = DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'],
                    result = DeepFace.analyze(img_path=frame, actions=['emotion'],
                                              enforce_detection=False,
                                              detector_backend="opencv",
                                              align=True,
                                              silent=False)
                    result = result[0]

                    # Draw bounding box and emotion text
                    face_coordinates = result["region"]
                    x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    emotion_text = f"{result['dominant_emotion']}"
                    cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    # texts = [
                    #     f"Age: {result['age']}",
                    #     f"Gender: {result['dominant_gender']}",
                    #     f"Race: {result['dominant_race']}",
                    #     f"Emotion: {result['dominant_emotion']}",
                    # ]
                    texts = [
                        # f"Age: {result['age']}",
                        # f"Gender: {result['dominant_gender']}",
                        # f"Race: {result['dominant_race']}",
                        f"Dominant Emotion: {result['dominant_emotion']} {round(result['emotion'][result['dominant_emotion']], 1)}",
                    ]
                    frame = overlay_text_on_frame(frame, texts)
                except Exception as e:
                    st.error(f"Error in analyzing frame: {e}")

                # Create a blank canvas to combine the frames
                combined_frame = np.zeros((720, 640, 3), dtype=np.uint8)

                # Place student frame
                combined_frame[0:240, 160:480] = frame

                # Encode frame as JPEG to reduce size
                _, buffer = cv2.imencode('.jpg', combined_frame)
                data = {
                    "frame": buffer.tobytes(),
                    "texts": texts
                }

                try:
                    await websocket.send(pickle.dumps(data))
                except Exception as e:
                    st.error(f"Error in sending data: {e}")
                    break

                # Display combined frame
                combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                stframe.image(combined_frame_rgb, channels='RGB', caption='Combined Stream', use_column_width=True)

                # Check stop checkbox state
                stop = st.session_state.stop_checkbox

                # 추가적인 대기 시간 조정
                await asyncio.sleep(0.1)

            cap.release()
    except Exception as e:
        st.error(f"Connection error: {e}")

def main():
    # Initialize the session state for the checkbox
    if 'stop_checkbox' not in st.session_state:
        st.session_state.stop_checkbox = False

    asyncio.run(send_video())

if __name__ == "__main__":
    main()