import uvicorn
import socketio
import asyncio
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from frame_processing.multi_camera_processor import MultiCameraProcessor
from frame_processing.config import Config
from calibration.intrinsic_calibration import IntrinsicCameraStreamer
from calibration.extrinsic_calibration import ExtrinsicCameraStreamer
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


@api_app.get("/intrinsic-images-preview")
def get_intrinsic_images():
    return sio.intrinsic_camera_streamer.get_saved_images_base64()


@api_app.get("/intrinsic-camera-calibration")
def calibrate_intrinsic():
    return sio.intrinsic_camera_streamer.calibrate_intrinsic_from_folder()


@api_app.get("/extrinsic-images-preview")
def get_extrinsic_images():
    return sio.extrinsic_camera_streamer.get_saved_images_base64()


@api_app.post("/extrinsic-camera-calibration")
async def calibrate_extrinsic(body=Body(...)):
    """
    Accepts a list of intrinsic parameters and performs extrinsic calibration.
    """

    # Convert intrinsics list to a dict: {index: {"K": ..., "distCoef": ...}}
    intrinsics_dict = {
        cam['index']: {
            'K': cam['K'],
            'distCoef': cam['distCoef']
        }
        for cam in body["intrinsics"]
    }

    result = sio.extrinsic_camera_streamer.calibrate_all_extrinsics(
        intrinsics_dict)
    return result


# ----------------------------------------------------
# Socket.IO setup
# ----------------------------------------------------
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=["http://localhost:8080"]
)

sio.multi_camera_tracker = None
sio.intrinsic_camera_streamer = None
sio.extrinsic_camera_streamer = None
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

    if sio.intrinsic_camera_streamer:
        sio.intrinsic_camera_streamer.stop()

    if sio.extrinsic_camera_streamer:
        sio.extrinsic_camera_streamer.stop()


@sio.event
async def start(sid, data):
    camera_indexes = [camera['index'] for camera in data["cameras"]]

    sio.multi_camera_tracker = MultiCameraProcessor(
        camera_indexes=camera_indexes,
        sio=sio,
        calibration_data=data["calibrationData"]
    )

    asyncio.create_task(sio.multi_camera_tracker.run())


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


@sio.on('start-intrinsic-calibration')
def start_intrinsic_calibration(sid, data):
    sio.intrinsic_camera_streamer = IntrinsicCameraStreamer(
        sio, data.get('camera_index'))
    asyncio.create_task(sio.intrinsic_camera_streamer.start())


@sio.on('intrinsic-request-frame-save')
def handle_intrinsic_save_frame(sid, data):
    Config.INTRINSIC_FRAME_NUMBER_TO_SAVE = data.get('frame_number')
    Config.INTRINSIC_FRAME_REQUESTED = True


@sio.on('start-extrinsic-calibration')
def start_extrinsic_calibration(sid, data):
    print(data.get("camera_indexes"))
    sio.extrinsic_camera_streamer = ExtrinsicCameraStreamer(
        sio, data.get('camera_indexes'))
    asyncio.create_task(sio.extrinsic_camera_streamer.start())


@sio.on('extrinsic-request-frame-save')
def handle_extrinsic_save_frame(sid, data):
    Config.EXTRINSIC_FRAME_NUMBER_TO_SAVE = data.get('frame_number')
    Config.EXTRINSIC_FRAME_REQUESTED = True

# ----------------------------------------------------
# Entry point
# ----------------------------------------------------


if __name__ == "__main__":
    uvicorn.run("main:main_app", host="0.0.0.0", port=8000, reload=True)
