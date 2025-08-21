from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import cv2
import socketio
import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from calibration.extrinsic_calibration import ExtrinsicCameraStreamer
from calibration.intrinsic_calibration import IntrinsicCameraStreamer
from frame_processing.config import Config
from frame_processing.multi_camera_processor import MultiCameraProcessor


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
def read_root() -> Dict[str, str]:
    return {"message": "Hello from FastAPI + Socket.IO"}


def get_available_cameras(max_index: int = 10) -> List[int]:
    """Probe indices [0..max_index) and return those that open successfully."""
    available: List[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            available.append(index)
            cap.release()
    return available


@api_app.get("/available-cameras", response_model=List[int])
def available_cameras() -> List[int]:
    return get_available_cameras()


@api_app.get("/intrinsic-images-preview")
def get_intrinsic_images() -> List[str]:
    return sio.intrinsic_camera_streamer.get_saved_images_base64()


@api_app.get("/intrinsic-camera-calibration")
def calibrate_intrinsic() -> Dict[str, Any]:
    return sio.intrinsic_camera_streamer.calibrate_intrinsic_from_folder()


@api_app.get("/extrinsic-images-preview")
def get_extrinsic_images() -> Dict[str, List[str]]:
    return sio.extrinsic_camera_streamer.get_saved_images_base64()


@api_app.post("/extrinsic-camera-calibration")
async def calibrate_extrinsic(body: Dict[str, Any] = Body(...)) -> Dict[int, Optional[Dict[str, List[List[float]]]]]:
    """
    Accept a list of intrinsic parameters and perform extrinsic calibration.

    Expected body format:
      {
        "intrinsics": [
          {"index": 0, "K": [[...]*3]*3, "dist_coef" OR "distCoef": [...]},
          {"index": 1, "K": [[...]*3]*3, "dist_coef" OR "distCoef": [...]},
          ...
        ]
      }
    """
    intrinsics_list: List[Dict[str, Any]] = body["intrinsics"]

    # Normalize into {index: {"K": ..., "dist_coef": ...}}
    intrinsics_dict: Dict[int, Dict[str, Any]] = {}
    for cam in intrinsics_list:
        idx = int(cam["index"])
        dist = cam.get("dist_coef", cam.get("distCoef", []))
        intrinsics_dict[idx] = {
            "K": cam["K"],
            "dist_coef": dist,
        }

    result = sio.extrinsic_camera_streamer.calibrate_all_extrinsics(
        intrinsics_dict)
    return result


# ----------------------------------------------------
# Socket.IO setup
# ----------------------------------------------------

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=["http://localhost:8080"],
)

# Will be attached at runtime
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
async def connect(sid: str, environ: Dict[str, Any]) -> None:
    print(f"Client connected: {sid}")


@sio.event
async def disconnect(sid: str) -> None:
    print(f"Client disconnected: {sid}")
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.stop()

    if sio.intrinsic_camera_streamer:
        sio.intrinsic_camera_streamer.stop()

    if sio.extrinsic_camera_streamer:
        sio.extrinsic_camera_streamer.stop()


@sio.event
async def start(sid: str, data: Dict[str, Any]) -> None:
    camera_indexes = [int(camera["index"]) for camera in data["cameras"]]

    sio.multi_camera_tracker = MultiCameraProcessor(
        camera_indexes=camera_indexes,
        sio=sio,
        calibration_data=data["calibrationData"],
    )

    asyncio.create_task(sio.multi_camera_tracker.run())


@sio.event
async def pause(sid: str) -> None:
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.pause()


@sio.event
async def stop(sid: str) -> None:
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.stop()


@sio.event
async def resume(sid: str) -> None:
    if sio.multi_camera_tracker:
        sio.multi_camera_tracker.resume()


@sio.event
async def fullscreen(sid: str, camera_id: str) -> None:
    Config.FULLSCREEN_CAMERA = camera_id
    Config.RESIZE_SHAPE_FOR_FRONTEND = (1280, 720)


@sio.event
async def exit_fullscreen(sid: str) -> None:
    Config.FULLSCREEN_CAMERA = None
    Config.RESIZE_SHAPE_FOR_FRONTEND = Config.RESIZE_SHAPE


@sio.event
async def reset(sid: str) -> None:
    if sio.multi_camera_tracker:
        asyncio.create_task(sio.multi_camera_tracker.reset())


@sio.on("start-intrinsic-calibration")
def start_intrinsic_calibration(sid: str, data: Dict[str, Any]) -> None:
    sio.intrinsic_camera_streamer = IntrinsicCameraStreamer(
        sio, int(data.get("camera_index"))
    )
    asyncio.create_task(sio.intrinsic_camera_streamer.start())


@sio.on("intrinsic-request-frame-save")
def handle_intrinsic_save_frame(sid: str, data: Dict[str, Any]) -> None:
    Config.INTRINSIC_FRAME_NUMBER_TO_SAVE = int(data.get("frame_number"))
    Config.INTRINSIC_FRAME_REQUESTED = True


@sio.on("start-extrinsic-calibration")
def start_extrinsic_calibration(sid: str, data: Dict[str, Any]) -> None:
    print(data.get("camera_indexes"))
    sio.extrinsic_camera_streamer = ExtrinsicCameraStreamer(
        sio, list(map(int, data.get("camera_indexes", [])))
    )
    asyncio.create_task(sio.extrinsic_camera_streamer.start())


@sio.on("extrinsic-request-frame-save")
def handle_extrinsic_save_frame(sid: str, data: Dict[str, Any]) -> None:
    Config.EXTRINSIC_FRAME_NUMBER_TO_SAVE = int(data.get("frame_number"))
    Config.EXTRINSIC_FRAME_REQUESTED = True


# ----------------------------------------------------
# Entry point
# ----------------------------------------------------

if __name__ == "__main__":
    # Run the ASGI app: uvicorn looks up "main_app" in this module.
    uvicorn.run("main:main_app", host="0.0.0.0", port=8000, reload=True)
