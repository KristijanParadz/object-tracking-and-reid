import re
from typing import Callable
import base64
import cv2


def make_frame_encoder(fmt: str = '.jpg') -> Callable[[any], str | None]:
    def encoder(frame):
        ret, buffer = cv2.imencode(fmt, frame)
        if not ret:
            return None
        return base64.b64encode(buffer).decode('utf-8')
    return encoder


def make_frame_number_extractor() -> Callable[[str], int]:
    pattern = re.compile(r'frame_(\d+)\.jpg')

    def extractor(path: str) -> int:
        match = pattern.search(path)
        return int(match.group(1)) if match else -1

    return extractor
