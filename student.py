import cv2
import asyncio
import websockets
import streamlit as st
from PIL import Image
import pickle
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
import io

# 서버 주소 설정
WEBSOCKET_HOST = '127.0.0.1'
WEBSOCKET_PORT = 9998

# 시간에 따른 감정 기록을 저장할 리스트
emotion_history = []
time_history = []
emotion_values_history = []
start_time_offset = 0

# 감정의 순서를 정의 (부정적 -> 긍정적)
emotion_order = ['angry', 'disgust', 'fear', 'sad', 'neutral', 'happy', 'surprise']

# 부정적 감정 목록
negative_emotions = ['angry', 'disgust', 'fear', 'sad']

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

def calculate_y_position(emotion, value):
    idx = emotion_order.index(emotion)
    if idx == len(emotion_order) - 1:
        return idx  # 마지막 감정인 경우, 바로 위치

    # 감정 간의 위치를 점수에 따라 계산
    return idx + (value / 100.0)

def plot_emotion_history(height, x_start):
    plt.figure(figsize=(6, height / 100))  # 세로 크기 맞추기 위해 플롯의 height 조정

    # X, Y 좌표 배열 초기화
    x_values = []
    y_values = []

    # 각 시간에 따른 감정과 해당 위치 결정
    for i in range(len(emotion_history)):
        emotion = emotion_history[i]
        y_pos = calculate_y_position(emotion, emotion_values_history[i][emotion])

        x_values.append(time_history[i] + x_start)
        y_values.append(y_pos)

    # 선 그래프 그리기 (항상 파란색)
    plt.plot(x_values, y_values, color='blue')

    plt.yticks(ticks=range(len(emotion_order)), labels=emotion_order)
    plt.ylim(-0.5, len(emotion_order) - 0.5)
    plt.xlim(x_start, x_start + 60)  # 1분 (60초) 동안의 범위
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion')
    plt.title('Emotion Over Time')
    plt.grid(True)

    # 플롯 이미지를 바이트 배열로 변환
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # 이미지를 OpenCV 형식으로 변환
    plot_img = Image.open(buf)
    plot_img = np.array(plot_img)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    return plot_img

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
                    result = DeepFace.analyze(img_path=frame, actions=['emotion'],
                                              enforce_detection=False,
                                              detector_backend="opencv",
                                              align=True,
                                              silent=True)  # silent=True로 경고 출력 방지
                    result = result[0]

                    # Emotion analysis text
                    dominant_emotion = result['dominant_emotion']
                    neutral_value = result['emotion'].get('neutral', 100)
                    texts = [
                        f"Dominant Emotion: {dominant_emotion} {round(result['emotion'][dominant_emotion], 1)}%",
                    ]

                    # Determine border color based on emotion
                    if (dominant_emotion == "neutral" and neutral_value < 30) or dominant_emotion in negative_emotions:
                        border_color = (0, 0, 255)  # Red border for negative or low neutral emotion
                    else:
                        border_color = (0, 255, 0)  # Green border for positive or high neutral emotion

                    # Update emotion and time history
                    emotion_history.append(dominant_emotion)
                    time_history.append(elapsed_time_seconds)
                    emotion_values_history.append(result['emotion'])

                    # Overlay text and border on the frame
                    frame = overlay_text_on_frame(frame, texts, border_color)

                    # Plot the emotion history
                    plot_img = plot_emotion_history(frame.shape[0], start_time_offset)  # 프레임 높이에 맞춰 플롯 생성

                    # Resize plot_img to match the frame height
                    plot_img = cv2.resize(plot_img, (int(plot_img.shape[1] * frame.shape[0] / plot_img.shape[0]), frame.shape[0]))

                    # Combine student frame and plot with a gap
                    gap = 30  # 학생 프레임과 플롯 사이에 추가 공간
                    combined_frame = np.hstack((frame, np.zeros((frame.shape[0], gap, 3), dtype=np.uint8), plot_img))

                    # Display combined frame (use placeholder to update)
                    combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(combined_frame_rgb, channels='RGB', caption='Combined Stream', use_column_width=True)

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

def main():
    # Initialize the session state for the checkbox
    if 'stop_checkbox' not in st.session_state:
        st.session_state.stop_checkbox = False

    asyncio.run(send_video())

if __name__ == "__main__":
    main()