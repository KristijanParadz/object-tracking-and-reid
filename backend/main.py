import uvicorn
import socketio
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from frame_processing.multi_camera_processor import MultiCameraProcessor
from frame_processing.config import Config
from typing import List
import cv2

# ----------------------------------------------------
# FastAPI app for API endpoints
# ----------------------------------------------------

api_app = FastAPI()

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api_app.get("/")
def read_root():
    return {"message": "Hello from FastAPI + Socket.IO"}


def get_available_cameras(max_index: int = 10) -> List[int]:
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            available.append(index)
            cap.release()
    return available


@api_app.get("/available-cameras", response_model=List[int])
def available_cameras():
    return get_available_cameras()


# ----------------------------------------------------
# Socket.IO setup
# ----------------------------------------------------

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=["http://localhost:8080"]
)

sio.multi_camera_tracker = None
socketio_app = socketio.ASGIApp(sio, socketio_path="socket.io")


# ----------------------------------------------------
# Main FastAPI app (used as ASGI app entrypoint)
# ----------------------------------------------------

main_app = FastAPI()


# Mount API and Socket.IO apps
main_app.mount("/api", api_app)
main_app.mount("/socket.io", socketio_app)


# ----------------------------------------------------
# Socket.IO event handlers
# ----------------------------------------------------

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.stop()


@sio.event
async def start(sid, data):
    video_paths = [
        'videos/hd_00_00.mp4',
        # 'videos/hd_00_01.mp4',
        # 'videos/hd_00_02.mp4',
        'videos/hd_00_03.mp4'
    ]

    sio.multi_camera_tracker = MultiCameraProcessor(
        video_paths=video_paths,
        sio=sio
    )

    print(data)

    # asyncio.create_task(sio.multi_camera_tracker.run())


@sio.event
async def pause(sid):
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.pause()


@sio.event
async def stop(sid):
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.stop()


@sio.event
async def resume(sid):
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.resume()


@sio.event
async def fullscreen(sid, camera_id):
    Config.FULLSCREEN_CAMERA = camera_id
    Config.RESIZE_SHAPE_FOR_FRONTEND = (1280, 720)


@sio.event
async def exit_fullscreen(sid):
    Config.FULLSCREEN_CAMERA = None
    Config.RESIZE_SHAPE_FOR_FRONTEND = Config.RESIZE_SHAPE


@sio.event
async def reset(sid):
    if sio.multi_camera_tracker:
        asyncio.create_task(sio.multi_camera_tracker.reset())


# ----------------------------------------------------
# Entry point
# ----------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:main_app", host="0.0.0.0", port=8000, reload=True)
