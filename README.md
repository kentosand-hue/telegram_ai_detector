# Telegram AI Detector (MVP)

MVP Telegram-бот для приема изображений и видео и выдачи вероятности AI-генерации.
По умолчанию используется локальная ConvNeXtV2 модель.

## Быстрый старт

1. Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Создайте файл `.env` на основе примера и укажите токен бота:

```bash
copy .env.example .env
```
3. Запустите бота:

```bash
python run.py
```

## Конфигурация

- `TELEGRAM_TOKEN` — токен вашего бота
- `TELEGRAM_PROXY` — прокси для Telegram API (опционально)
- `TELEGRAM_TIMEOUT_SEC` — таймаут запросов к Telegram API
- `MESSAGE_THEME` — стиль сообщений: `emotional` или `minimal`
- `MAX_VIDEO_FRAMES` — максимум анализируемых кадров из видео
- `FRAME_EVERY_SEC` — интервал выборки кадров
- `HISTORY_LIMIT` — количество записей в команде `/history`
- `MAX_FILE_SIZE_MB` — максимальный размер входного файла
- `TELEGRAM_DOWNLOAD_LIMIT_MB` — ограничение загрузки Bot API (обычно 20 МБ)
- `MODEL_BACKEND` — `convnext` (локальный ConvNeXt) или `hf` (Hugging Face pipeline)
- `MODEL_CHECKPOINT` — путь к `checkpoint_phase2.pth` для `convnext`
- `CONVNEXT_MODEL_NAME` — имя модели в timm (по умолчанию `convnextv2_base.fcmae_ft_in1k`)
- `MODEL_NAME` — модель для `hf` backend (задается вручную)
- `MODEL_DEVICE` — устройство для инференса: `auto`, `cpu`, `cuda`

## Message Theme

- `MESSAGE_THEME=emotional` — сообщения с эмодзи (по умолчанию)
- `MESSAGE_THEME=minimal` — минималистичные сообщения без эмодзи

## Примечания

- ConvNeXt backend требует локальный чекпойнт, автоматически он не скачивается.
- Для быстрого старта можно установить `MODEL_BACKEND=hf`.
- Результаты не являются 100% точными, используйте как вспомогательную проверку.
