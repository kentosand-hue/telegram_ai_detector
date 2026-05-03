from __future__ import annotations

from pathlib import Path
from typing import Iterable

from aiogram import Bot


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


async def download_telegram_file(bot: Bot, file_id: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    file = await bot.get_file(file_id)
    await bot.download_file(file.file_path, destination=dest_path)


def read_bytes(file_path: Path) -> bytes:
    return file_path.read_bytes()


def extract_video_frames_bytes(
    video_path: Path,
    *,
    every_sec: float,
    max_frames: int,
) -> list[bytes]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("OpenCV is required for video analysis") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Не удалось открыть видеофайл")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frame_interval = max(int(round(fps * every_sec)), 1)
    frames: list[bytes] = []
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if index % frame_interval == 0:
            ok_enc, buffer = cv2.imencode(".jpg", frame)
            if ok_enc:
                frames.append(buffer.tobytes())
                if len(frames) >= max_frames:
                    break
        index += 1

    cap.release()
    return frames
