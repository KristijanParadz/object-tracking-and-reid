import re
from typing import Callable
import base64
import cv2


def make_frame_encoder(fmt: str = '.jpg') -> Callable[[any], str | None]:
    """
    Create a function that encodes image frames into Base64 strings.

    Args:
        fmt (str, optional): Image format for encoding (default: '.jpg').

    Returns:
        Callable[[any], str | None]: A function that takes a frame (NumPy array),
        encodes it into the specified image format, and returns the Base64-encoded
        string. Returns None if encoding fails.
    """
    def encoder(frame):
        ret, buffer = cv2.imencode(fmt, frame)
        if not ret:
            return None
        return base64.b64encode(buffer).decode('utf-8')
    return encoder


def make_frame_number_extractor() -> Callable[[str], int]:
    """
    Create a function that extracts frame numbers from file paths.

    The extractor looks for filenames matching the pattern "frame_<number>.jpg".

    Returns:
        Callable[[str], int]: A function that takes a file path string and returns
        the extracted frame number as an integer. If no number is found, returns -1.
    """
    pattern = re.compile(r'frame_(\d+)\.jpg')

    def extractor(path: str) -> int:
        match = pattern.search(path)
        return int(match.group(1)) if match else -1

    return extractor
