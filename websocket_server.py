import asyncio
import websockets
import pickle
from dotenv import main as dotmain
import os

dotmain.load_dotenv('environments.env')

# 서버 설정
WEBSOCKET_HOST = os.getenv('WEBSOCKET_HOST')
WEBSOCKET_PORT = int(os.getenv('WEBSOCKET_SERVER_PORT'))

connected_clients = set()

async def handle_connection(websocket, path):
    global connected_clients
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            data = pickle.loads(message)

            # 연결된 다른 클라이언트에게 데이터 브로드캐스트
            if data:
                await asyncio.wait([client.send(message) for client in connected_clients if client != websocket])

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")
    finally:
        connected_clients.remove(websocket)

async def start_server():
    server = await websockets.serve(handle_connection, WEBSOCKET_HOST, WEBSOCKET_PORT)
    print(f"WebSocket server started on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(start_server())