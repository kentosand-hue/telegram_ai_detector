from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv, dotenv_values


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TMP_DIR = ROOT_DIR / "tmp"


@dataclass(frozen=True)
class Settings:
    token: str
    db_path: Path
    tmp_dir: Path
    max_video_frames: int
    frame_every_sec: float
    history_limit: int
    max_file_size_mb: int
    telegram_download_limit_mb: int
    admin_chat_id: int | None
    telegram_proxy: str | None
    telegram_timeout_sec: float
    message_theme: str

def load_settings() -> Settings:
    # Load .env explicitly from project root to avoid cwd issues.
    env_path = ROOT_DIR / ".env"
    load_dotenv(env_path)
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        # Fallback for UTF-8 BOM or odd parsing edge-cases.
        values = dotenv_values(env_path)
        for key, value in values.items():
            if key and key.lstrip("\ufeff") == "TELEGRAM_TOKEN":
                token = (value or "").strip()
                break
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN is not set")

    max_video_frames = int(os.getenv("MAX_VIDEO_FRAMES", "10"))
    frame_every_sec = float(os.getenv("FRAME_EVERY_SEC", "1.0"))
    history_limit = int(os.getenv("HISTORY_LIMIT", "10"))
    max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    telegram_download_limit_mb = int(os.getenv("TELEGRAM_DOWNLOAD_LIMIT_MB", "20"))
    admin_chat_raw = os.getenv("ADMIN_CHAT_ID", "").strip()
    admin_chat_id = int(admin_chat_raw) if admin_chat_raw else None
    proxy = os.getenv("TELEGRAM_PROXY", "").strip()
    if not proxy:
        proxy = os.getenv("HTTPS_PROXY", "").strip() or os.getenv("HTTP_PROXY", "").strip()
    proxy = proxy or None
    telegram_timeout_sec = float(os.getenv("TELEGRAM_TIMEOUT_SEC", "30"))
    message_theme = os.getenv("MESSAGE_THEME", "emotional").strip().lower() or "emotional"
    if message_theme not in {"emotional", "minimal"}:
        raise RuntimeError("MESSAGE_THEME must be 'emotional' or 'minimal'")
    return Settings(
        token=token,
        db_path=DATA_DIR / "checks.db",
        tmp_dir=TMP_DIR,
        max_video_frames=max_video_frames,
        frame_every_sec=frame_every_sec,
        history_limit=history_limit,
        max_file_size_mb=max_file_size_mb,
        telegram_download_limit_mb=telegram_download_limit_mb,
        admin_chat_id=admin_chat_id,
        telegram_proxy=proxy,
        telegram_timeout_sec=telegram_timeout_sec,
        message_theme=message_theme,
    )


