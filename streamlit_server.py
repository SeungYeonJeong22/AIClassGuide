import cv2
import asyncio
import websockets
import streamlit as st
from PIL import Image
import pickle
import numpy as np

# 서버 설정
STREAMLIT_HOST = '0.0.0.0'
STREAMLIT_PORT = 8502  # 스트림릿 서버 포트 변경

async def receive_video(websocket, path):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

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
            student_frame_resized = cv2.resize(student_frame, (480, 360))  # 학생 프레임을 더 크게 조정
            teacher_frame_resized = cv2.resize(teacher_frame, (640, 360))  # 교사 프레임 크기 수정

            # Create a blank canvas to combine the frames
            combined_frame = np.zeros((720, 640, 3), dtype=np.uint8)

            # Place teacher frame
            combined_frame[360:720, 0:640] = teacher_frame_resized

            # Place student frame
            combined_frame[0:360, 80:560] = student_frame_resized  # 위치 조정

            # Convert OpenCV image to PIL image
            combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            combined_frame_pil = Image.fromarray(combined_frame_rgb)

            # Display the combined frame
            stframe.image(combined_frame_pil, caption='Combined Stream', use_column_width=True)

    except websockets.exceptions.ConnectionClosed as e:
        st.error(f"Connection closed: {e}")
    finally:
        cap.release()

async def start_server():
    async with websockets.serve(receive_video, STREAMLIT_HOST, STREAMLIT_PORT):
        await asyncio.Future()  # run forever

def main():
    st.title("Teacher's Live Stream")
    st.write("Waiting for student to join...")

    asyncio.run(start_server())

if __name__ == "__main__":
    main()