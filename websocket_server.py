import cv2
import asyncio
import websockets
import pickle
import numpy as np

# 서버 설정
WEBSOCKET_HOST = '0.0.0.0'
WEBSOCKET_PORT = 9999

async def receive_video(websocket, path):
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
            student_frame_resized = cv2.resize(student_frame, (960, 540))  # 학생 프레임을 더 크게 조정
            teacher_frame_resized = cv2.resize(teacher_frame, (1280, 720))  # 교사 프레임 크기 수정

            # Create a blank canvas to combine the frames
            combined_frame = np.zeros((1260, 1280, 3), dtype=np.uint8)  # 세로 길이를 줄임

            # Place teacher frame
            combined_frame[540:1260, 0:1280] = teacher_frame_resized

            # Place student frame
            combined_frame[0:540, 160:1120] = student_frame_resized  # 위치 조정

            cv2.imshow("Combined Frame", combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def main():
    async with websockets.serve(receive_video, WEBSOCKET_HOST, WEBSOCKET_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())