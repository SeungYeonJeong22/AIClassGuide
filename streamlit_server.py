import cv2
import asyncio
import websockets
import streamlit as st
from PIL import Image
import pickle
import numpy as np

# 서버 설정
STREAMLIT_HOST = '0.0.0.0'
STREAMLIT_PORT = 8502  # 스트림릿 서버 포트
WEBSOCKET_PORT = 9998  # WebSocket 포트

st.title("Teacher's Live Stream")  # 페이지의 최상단에 고정
status_text = st.empty()
stframe = st.empty()

async def handle_client(websocket, path):
    global stframe, status_text
    cap = cv2.VideoCapture(0)
    stframe.empty()  # Clear the Streamlit frame
    status_text.empty()  # Clear the status text

    try:
        while True:
            message = await websocket.recv()
            data = pickle.loads(message)
            frame_data = data["frame"]

            # np.frombuffer를 사용하여 바이트 데이터를 numpy 배열로 변환
            nparr = np.frombuffer(frame_data, np.uint8)
            student_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            ret, teacher_frame = cap.read()
            if not ret:
                break

            # Resize frames
            student_frame_resized = cv2.resize(student_frame, (960, 540))  # 학생 프레임을 더 크게 조정
            teacher_frame_resized = cv2.resize(teacher_frame, (1280, 720))  # 교사 프레임 크기 수정

            # Create a blank canvas to combine the frames
            combined_frame = np.zeros((1260, 1280, 3), dtype=np.uint8)  # 세로 길이를 줄임

            # Place teacher frame
            combined_frame[540:1260, 0:1280] = teacher_frame_resized

            # Place student frame
            combined_frame[0:540, 160:1120] = student_frame_resized  # 위치 조정

            # Convert OpenCV image to PIL image
            combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            combined_frame_pil = Image.fromarray(combined_frame_rgb)

            # Display the combined frame
            stframe.image(combined_frame_pil, caption='Combined Stream', use_column_width=True)

    except websockets.exceptions.ConnectionClosed as e:
        stframe.empty()
        status_text.warning("Waiting for student to join...")
        print(f"Connection closed: {e}")
    finally:
        cap.release()

async def start_server():
    async with websockets.serve(handle_client, STREAMLIT_HOST, WEBSOCKET_PORT):
        await asyncio.Future()  # run forever

def main():
    global status_text
    status_text.warning("Waiting for student to join...")

    asyncio.run(start_server())

if __name__ == "__main__":
    main()