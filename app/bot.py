from __future__ import annotations

import logging
import re
from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Document, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.config import DATA_DIR, TMP_DIR, load_settings
from app.db import get_history, init_db, save_check
from app.detector import (
    explanation_from_probability,
    label_from_probability,
    probability_from_bytes,
    visual_signs_from_bytes,
    visual_signs_from_frames,
)
from app.media import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    download_telegram_file,
    extract_video_frames_bytes,
    read_bytes,
)


logger = logging.getLogger(__name__)
router = Router()
SETTINGS = None

_EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🛠️ Помощь", callback_data="menu:help"),
                InlineKeyboardButton(text="ℹ️ О боте", callback_data="menu:about"),
            ],
            [
                InlineKeyboardButton(text="🗂️ История", callback_data="menu:history"),
            ],
        ]
    )


def _theme_text(text: str, settings) -> str:
    theme = getattr(settings, "message_theme", "emotional")
    if theme != "minimal":
        return text

    themed = _EMOJI_PATTERN.sub("", text)
    themed = themed.replace("• ", "- ")
    themed = re.sub(r"[ \t]{2,}", " ", themed)
    themed = "\n".join(line.strip() for line in themed.splitlines())
    themed = re.sub(r"\n{3,}", "\n\n", themed)
    return themed.strip()


async def _answer(message: Message, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> None:
    themed = _theme_text(text, SETTINGS)
    await message.answer(themed, parse_mode="HTML", reply_markup=reply_markup)


def _now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _max_file_size_bytes(settings) -> int:
    limit_mb = min(int(settings.max_file_size_mb), int(settings.telegram_download_limit_mb))
    return limit_mb * 1024 * 1024


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f}"


def _probability_bar(probability: float) -> str:
    filled = round(max(0.0, min(probability, 1.0)) * 10)
    return "█" * filled + "░" * (10 - filled)


def _is_too_large(file_size: int | None, settings) -> bool:
    if file_size is None:
        return False
    return file_size > _max_file_size_bytes(settings)


def _size_limit_message(file_size: int | None, settings) -> str:
    limit_mb = min(int(settings.max_file_size_mb), int(settings.telegram_download_limit_mb))
    source = "ограничение Telegram Bot API" if limit_mb == int(settings.telegram_download_limit_mb) else "настройка бота"
    if file_size is None:
        return (
            "<b>Файл слишком большой</b>\n\n"
            f"📏 Лимит: <b>{limit_mb} МБ</b>\n"
            f"Источник лимита: {escape(source)}"
        )
    return (
        "<b>Файл слишком большой</b>\n\n"
        f"📦 Размер файла: <b>{_format_mb(file_size)} МБ</b>\n"
        f"📏 Лимит: <b>{limit_mb} МБ</b>\n"
        f"Источник лимита: {escape(source)}"
    )


async def _notify_admin(
    *,
    message: Message,
    bot: Bot,
    settings,
    response_text: str | None = None,
    error_text: str | None = None,
    tag: str | None = None,
) -> None:
    admin_chat_id = getattr(settings, "admin_chat_id", None)
    if not admin_chat_id:
        return

    user = message.from_user
    username = f"@{user.username}" if user and user.username else "-"
    header = f"Пользователь: {user.id} {username}"
    if tag:
        header = f"{tag}\n{header}"

    try:
        await bot.forward_message(admin_chat_id, message.chat.id, message.message_id)
    except Exception:
        try:
            await bot.copy_message(admin_chat_id, message.chat.id, message.message_id)
        except Exception as exc:
            logger.warning("Admin notify forward/copy failed: %s", exc)

    parts = [header]
    if response_text:
        parts.append(f"Ответ:\n{response_text}")
    if error_text:
        parts.append(f"Ошибка:\n{error_text}")

    try:
        await bot.send_message(admin_chat_id, "\n".join(parts))
    except Exception as exc:
        logger.warning("Admin notify send failed: %s", exc)


def _is_image_document(document: Document) -> bool:
    if document.mime_type and document.mime_type.startswith("image/"):
        return True
    if document.file_name:
        return Path(document.file_name).suffix.lower() in IMAGE_EXTENSIONS
    return False


def _is_video_document(document: Document) -> bool:
    if document.mime_type and document.mime_type.startswith("video/"):
        return True
    if document.file_name:
        return Path(document.file_name).suffix.lower() in VIDEO_EXTENSIONS
    return False


def _format_result(file_name: str, probability: float, signs: list[str] | None = None) -> str:
    percent = probability * 100
    label = label_from_probability(probability)
    explanation = explanation_from_probability(probability)
    signs_text = ""
    if signs:
        signs_lines = "\n".join(f"• {escape(sign)}" for sign in signs)
        signs_text = f"\n\n<b>🧩 Визуальные признаки</b>\n{signs_lines}"

    return (
        "<b>✅ Анализ завершен</b>\n\n"
        f"📄 <b>Файл:</b> <code>{escape(file_name)}</code>\n"
        f"📊 <b>AI-вероятность:</b> <b>{percent:.2f}%</b>\n"
        f"<code>{_probability_bar(probability)}</code>\n\n"
        f"🧠 <b>Итог:</b>\n{escape(label)}\n\n"
        f"🔎 <b>Пояснение:</b>\n{escape(explanation)}"
        f"{signs_text}\n\n"
        "⚠️ <i>Это вероятностная оценка модели, а не окончательный вердикт.</i>"
    )


def _start_text() -> str:
    return (
        "<b>👋 AI Detector готов к работе</b>\n\n"
        "Отправьте изображение или видео, и я покажу:\n"
        "• 📊 вероятность AI-генерации\n"
        "• 🧠 итоговый статус\n"
        "• 🔎 краткое объяснение\n"
        "• 🧩 найденные визуальные признаки\n\n"
        "<b>Команды</b>\n"
        "/help — помощь\n"
        "/about — о модели\n"
        "/history — история проверок"
    )


def _help_text(settings) -> str:
    limit_mb = min(int(settings.max_file_size_mb), int(settings.telegram_download_limit_mb))
    return (
        "<b>🛠️ Как пользоваться</b>\n\n"
        "1. Отправьте фото или видео.\n"
        "2. Для фото лучше использовать отправку <b>как файл</b>, без сжатия.\n"
        "3. Получите процент, итог, пояснение и признаки.\n\n"
        "<b>Команды</b>\n"
        "/start — приветствие\n"
        "/help — подсказка\n"
        "/about — о модели\n"
        "/history — последние проверки\n\n"
        "<b>Форматы</b>\n"
        "JPG, JPEG, PNG, MP4, MOV, AVI\n\n"
        f"📏 Лимит размера: <b>{limit_mb} МБ</b>"
    )


def _about_text() -> str:
    return (
        "<b>ℹ️ О боте</b>\n\n"
        "Я оцениваю вероятность AI-генерации и отдельно показываю визуальные признаки:\n"
        "• контраст\n"
        "• резкость и границы\n"
        "• насыщенность\n"
        "• цветовой баланс\n"
        "• плотность данных\n\n"
        "⚠️ <i>Результат является вероятностной оценкой и не считается 100% доказательством.</i>"
    )


def _history_text(settings, user_id: int) -> str:
    records = get_history(settings.db_path, user_id=user_id, limit=settings.history_limit)
    if not records:
        return "<b>🗂️ История пока пустая</b>\n\nОтправьте первый файл для проверки."

    lines = [f"<b>🗂️ История проверок</b>\nПоследние {settings.history_limit}:"]
    for idx, record in enumerate(records, start=1):
        percent = record.probability * 100
        lines.append(
            f"\n<b>{idx}. {escape(record.file_type.upper())}</b>\n"
            f"🕒 {escape(record.created_at)}\n"
            f"📁 <code>{escape(record.file_name)}</code>\n"
            f"📊 AI: <b>{percent:.2f}%</b>\n"
            f"🧠 {escape(record.label)}"
        )

    return "\n".join(lines)


async def _analyze_and_reply_image(
    *,
    message: Message,
    bot: Bot,
    file_id: str,
    file_name: str,
    settings,
) -> None:
    tmp_path = settings.tmp_dir / f"{uuid4().hex}{Path(file_name).suffix.lower()}"

    try:
        await download_telegram_file(bot, file_id, tmp_path)
        payload = read_bytes(tmp_path)
        probability = probability_from_bytes(payload)
        signs = visual_signs_from_bytes(payload)
        label = label_from_probability(probability)
        save_check(
            settings.db_path,
            user_id=message.from_user.id,
            username=message.from_user.username,
            file_type="image",
            file_name=file_name,
            probability=probability,
            label=label,
            created_at=_now_string(),
        )
        result_text = _format_result(file_name, probability, signs)
        await _answer(message, result_text)
        await _notify_admin(
            message=message,
            bot=bot,
            settings=settings,
            response_text=result_text,
            tag="Изображение",
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


async def _analyze_and_reply_video(
    *,
    message: Message,
    bot: Bot,
    file_id: str,
    file_name: str,
    settings,
) -> None:
    tmp_path = settings.tmp_dir / f"{uuid4().hex}{Path(file_name).suffix.lower()}"

    try:
        await download_telegram_file(bot, file_id, tmp_path)
        frames = extract_video_frames_bytes(
            tmp_path,
            every_sec=settings.frame_every_sec,
            max_frames=settings.max_video_frames,
        )
        if not frames:
            raise RuntimeError("Не удалось извлечь кадры из видео")

        probabilities = [probability_from_bytes(frame) for frame in frames]
        probability = sum(probabilities) / len(probabilities)
        signs = visual_signs_from_frames(frames)
        label = label_from_probability(probability)
        save_check(
            settings.db_path,
            user_id=message.from_user.id,
            username=message.from_user.username,
            file_type="video",
            file_name=file_name,
            probability=probability,
            label=label,
            created_at=_now_string(),
        )
        result_text = _format_result(file_name, probability, signs)
        await _answer(message, result_text)
        await _notify_admin(
            message=message,
            bot=bot,
            settings=settings,
            response_text=result_text,
            tag="Видео",
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.message(Command("start"))
async def cmd_start(message: Message, bot: Bot) -> None:
    text = _start_text()
    await _answer(message, text, reply_markup=_main_menu_keyboard())
    await _notify_admin(message=message, bot=bot, settings=SETTINGS, response_text=text, tag="Команда /start")


@router.message(Command("help"))
async def cmd_help(message: Message, bot: Bot) -> None:
    settings = SETTINGS
    text = _help_text(settings)
    await _answer(message, text, reply_markup=_main_menu_keyboard())
    await _notify_admin(message=message, bot=bot, settings=SETTINGS, response_text=text, tag="Команда /help")


@router.message(Command("about"))
async def cmd_about(message: Message, bot: Bot) -> None:
    text = _about_text()
    await _answer(message, text, reply_markup=_main_menu_keyboard())
    await _notify_admin(message=message, bot=bot, settings=SETTINGS, response_text=text, tag="Команда /about")


@router.message(Command("history"))
async def cmd_history(message: Message, bot: Bot) -> None:
    settings = SETTINGS
    text = _history_text(settings, message.from_user.id)
    await _answer(message, text, reply_markup=_main_menu_keyboard())
    await _notify_admin(message=message, bot=bot, settings=settings, response_text=text, tag="Команда /history")


@router.callback_query(F.data == "menu:help")
async def callback_help(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.answer(
        _theme_text(_help_text(SETTINGS), SETTINGS),
        parse_mode="HTML",
        reply_markup=_main_menu_keyboard(),
    )


@router.callback_query(F.data == "menu:about")
async def callback_about(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.answer(
        _theme_text(_about_text(), SETTINGS),
        parse_mode="HTML",
        reply_markup=_main_menu_keyboard(),
    )


@router.callback_query(F.data == "menu:history")
async def callback_history(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.answer(
        _theme_text(_history_text(SETTINGS, callback.from_user.id), SETTINGS),
        parse_mode="HTML",
        reply_markup=_main_menu_keyboard(),
    )


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot) -> None:
    settings = SETTINGS
    photo = message.photo[-1]
    file_name = f"{photo.file_unique_id}.jpg"

    if _is_too_large(photo.file_size, settings):
        text = _size_limit_message(photo.file_size, settings)
        await _answer(message, text)
        await _notify_admin(message=message, bot=bot, settings=settings, response_text=text, tag="Файл слишком большой")
        return

    await _answer(message, "<b>⏳ Файл получен</b>\nЗапускаю анализ, это может занять несколько секунд...")
    try:
        await _analyze_and_reply_image(
            message=message,
            bot=bot,
            file_id=photo.file_id,
            file_name=file_name,
            settings=settings,
        )
    except Exception as exc:
        logger.exception("Image analysis failed")
        text = f"<b>❌ Не удалось обработать изображение</b>\n\nПричина: {escape(str(exc))}"
        await _answer(message, text)
        await _notify_admin(message=message, bot=bot, settings=settings, error_text=text, tag="Ошибка изображения")


@router.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    settings = SETTINGS
    document = message.document

    if _is_too_large(document.file_size, settings):
        text = _size_limit_message(document.file_size, settings)
        await _answer(message, text)
        await _notify_admin(message=message, bot=bot, settings=settings, response_text=text, tag="Файл слишком большой")
        return

    if _is_image_document(document):
        await _answer(message, "<b>⏳ Файл получен</b>\nЗапускаю анализ, это может занять несколько секунд...")
        try:
            await _analyze_and_reply_image(
                message=message,
                bot=bot,
                file_id=document.file_id,
                file_name=document.file_name or f"{document.file_unique_id}.jpg",
                settings=settings,
            )
        except Exception as exc:
            logger.exception("Image analysis failed")
            text = f"<b>❌ Не удалось обработать изображение</b>\n\nПричина: {escape(str(exc))}"
            await _answer(message, text)
            await _notify_admin(message=message, bot=bot, settings=settings, error_text=text, tag="Ошибка изображения")
        return

    if _is_video_document(document):
        await _answer(message, "<b>⏳ Файл получен</b>\nЗапускаю анализ, это может занять несколько секунд...")
        try:
            await _analyze_and_reply_video(
                message=message,
                bot=bot,
                file_id=document.file_id,
                file_name=document.file_name or f"{document.file_unique_id}.mp4",
                settings=settings,
            )
        except Exception as exc:
            logger.exception("Video analysis failed")
            text = f"<b>❌ Не удалось обработать видео</b>\n\nПричина: {escape(str(exc))}"
            await _answer(message, text)
            await _notify_admin(message=message, bot=bot, settings=settings, error_text=text, tag="Ошибка видео")
        return

    text = (
        "<b>🚫 Формат пока не поддерживается</b>\n\n"
        "Поддерживаемые форматы:\n"
        "JPG, JPEG, PNG, MP4, MOV, AVI"
    )
    await _answer(message, text)
    await _notify_admin(message=message, bot=bot, settings=settings, response_text=text, tag="Неподдерживаемый формат")


@router.message(F.video)
async def handle_video(message: Message, bot: Bot) -> None:
    settings = SETTINGS
    video = message.video
    file_name = video.file_name or f"{video.file_unique_id}.mp4"

    if _is_too_large(video.file_size, settings):
        text = _size_limit_message(video.file_size, settings)
        await _answer(message, text)
        await _notify_admin(message=message, bot=bot, settings=settings, response_text=text, tag="Файл слишком большой")
        return

    await _answer(message, "<b>⏳ Файл получен</b>\nЗапускаю анализ, это может занять несколько секунд...")
    try:
        await _analyze_and_reply_video(
            message=message,
            bot=bot,
            file_id=video.file_id,
            file_name=file_name,
            settings=settings,
        )
    except Exception as exc:
        logger.exception("Video analysis failed")
        text = f"<b>❌ Не удалось обработать видео</b>\n\nПричина: {escape(str(exc))}"
        await _answer(message, text)
        await _notify_admin(message=message, bot=bot, settings=settings, error_text=text, tag="Ошибка видео")


@router.message()
async def handle_unknown(message: Message, bot: Bot) -> None:
    text = (
        "<b>🤖 Я работаю с изображениями и видео</b>\n\n"
        "Отправьте файл для анализа или используйте /help."
    )
    await _answer(message, text, reply_markup=_main_menu_keyboard())
    await _notify_admin(message=message, bot=bot, settings=SETTINGS, response_text=text, tag="Неизвестное сообщение")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    global SETTINGS
    SETTINGS = load_settings()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    init_db(SETTINGS.db_path)

    session = AiohttpSession(proxy=SETTINGS.telegram_proxy, timeout=SETTINGS.telegram_timeout_sec)
    bot = Bot(token=SETTINGS.token, session=session)

    dp = Dispatcher()
    dp.include_router(router)

    dp.run_polling(bot)


if __name__ == "__main__":
    main()













