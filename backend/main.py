import uvicorn
import socketio
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from frame_processing.multi_camera_processor import MultiCameraProcessor
from frame_processing.config import Config

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


socket_app = socketio.ASGIApp(
    sio, other_asgi_app=app, socketio_path="socket.io"
)

sio.multi_camera_tracker = None


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI + Socket.IO"}


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
async def start(sid):
    """
    START event: instantiate MultiCameraProcessor and spawn its run() method as a background task.
    """
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

    asyncio.create_task(sio.multi_camera_tracker.run())


@sio.event
async def pause(sid):
    """
    PAUSE event: tell the tracker to stop processing frames (but not exit).
    """
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.pause()


@sio.event
async def stop(sid):
    """
    STOP event: tell the tracker to stop processing frames (and exit).
    """
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.stop()


@sio.event
async def resume(sid):
    """
    RESUME event: unpause the loop, letting it continue from where it left off.
    """
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
    """
    RESET event:
      1. Stop the current loop if it’s running.
      2. Reinitialize all trackers from frame=0.
      3. Start the run loop again in a fresh task.
    """
    if sio.multi_camera_tracker:
        asyncio.create_task(sio.multi_camera_tracker.reset())


if __name__ == "__main__":
    uvicorn.run("main:socket_app", host="0.0.0.0", port=8000, reload=True)
