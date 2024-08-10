import cv2
import asyncio
import websockets
import pickle
from emotion_analysis import analyze_emotion
from plotting import plot_emotion_history
import streamlit as st
import numpy as np

# 서버 주소 설정
WEBSOCKET_HOST = '192.168.0.187'
WEBSOCKET_PORT = 9998

# 시간에 따른 감정 기록을 저장할 리스트
emotion_history = []
time_history = []
emotion_values_history = []
start_time_offset = 0

async def send_video():
    global start_time_offset
    uri = f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}"
    try:
        async with websockets.connect(uri) as websocket:
            cap = cv2.VideoCapture(0)

            st.title("Student's Live Stream with Emotion Analysis")

            # Placeholder for the combined frame
            frame_placeholder = st.empty()

            start_time = None
            stop = st.checkbox('Stop', key='stop_checkbox')

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break

                if start_time is None:
                    start_time = cv2.getTickCount() / cv2.getTickFrequency()

                elapsed_time = (cv2.getTickCount() / cv2.getTickFrequency()) - start_time
                elapsed_time_seconds = int(elapsed_time)

                # 1분 단위로 X축 초기화
                if elapsed_time_seconds > 60:
                    start_time_offset += 60  # 새로운 1분 구간으로 넘어감
                    start_time = None  # 새로운 타이밍 시작
                    time_history.clear()
                    emotion_history.clear()
                    emotion_values_history.clear()

                # Resize frame to reduce size
                frame = cv2.resize(frame, (640, 480))  # 적절한 해상도 조정

                # Analyze the frame using DeepFace
                try:
                    result, texts, border_color = analyze_emotion(frame)

                    # Update emotion and time history
                    emotion_history.append(result['dominant_emotion'])
                    time_history.append(elapsed_time_seconds)
                    emotion_values_history.append(result['emotion'])

                    # Overlay text and border on the frame
                    frame = overlay_text_on_frame(frame, texts, border_color)

                    # Plot the emotion history
                    plot_img = plot_emotion_history(frame.shape[0], start_time_offset, emotion_history, time_history, emotion_values_history)

                    # Resize plot_img to match the frame height
                    plot_img = cv2.resize(plot_img, (int(plot_img.shape[1] * frame.shape[0] / plot_img.shape[0]), frame.shape[0]))

                    # Combine student frame and plot with a gap
                    gap = 30  # 학생 프레임과 플롯 사이에 추가 공간
                    combined_frame = np.hstack((frame, np.zeros((frame.shape[0], gap, 3), dtype=np.uint8), plot_img))

                    # Display combined frame (use placeholder to update)
                    combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(combined_frame_rgb, channels='RGB', caption='Stream', use_column_width=True)

                    # Encode combined frame as JPEG to reduce size
                    _, buffer = cv2.imencode('.jpg', combined_frame)
                    data = {
                        "frame": buffer.tobytes()
                    }

                except Exception as e:
                    st.error(f"Error in analyzing frame: {e}")

                try:
                    await websocket.send(pickle.dumps(data))
                except Exception as e:
                    st.error(f"Error in sending data: {e}")
                    break

                # Check stop checkbox state
                stop = st.session_state.stop_checkbox

                # 추가적인 대기 시간 조정
                await asyncio.sleep(0.1)

            cap.release()

    except Exception as e:
        st.error(f"Connection error: {e}")

def overlay_text_on_frame(frame, texts, border_color):
    overlay = frame.copy()
    alpha = 0.6  # Adjust the transparency of the overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # White rectangle
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 50  # Where the first text is put into the overlay
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        text_position += 50

    # Draw a border around the frame
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), border_color, 10)

    return frame