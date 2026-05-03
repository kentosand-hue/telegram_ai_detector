from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _device_pref() -> str:
    return os.getenv("MODEL_DEVICE", "auto").strip().lower()


def _resolve_torch_device():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("PyTorch is required for model inference") from exc

    pref = _device_pref()
    if pref in {"cpu", "-1"}:
        return torch.device("cpu")
    if pref in {"cuda", "gpu", "0"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_pipeline_device() -> int:
    return 0 if _resolve_torch_device().type == "cuda" else -1


def _backend() -> str:
    return os.getenv("MODEL_BACKEND", "convnext").strip().lower()


def _checkpoint_path() -> Path:
    return Path(os.getenv("MODEL_CHECKPOINT", "models/convnext/checkpoint_phase2.pth"))


def _convnext_model_name() -> str:
    return os.getenv("CONVNEXT_MODEL_NAME", "convnextv2_base.fcmae_ft_in1k").strip() or "convnextv2_base.fcmae_ft_in1k"


def _model_name() -> str:
    name = os.getenv("MODEL_NAME", "").strip()
    if not name:
        raise RuntimeError("MODEL_NAME is not set for hf backend")
    return name


@lru_cache(maxsize=1)
def _get_pipeline():
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("transformers is required for model inference") from exc

    device = _resolve_pipeline_device()
    return pipeline("image-classification", model=_model_name(), device=device)


def _ai_probability_from_output(outputs: list[dict[str, Any]]) -> float:
    ai_score: float | None = None
    real_score: float | None = None

    for item in outputs:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if "fake" in label or "ai" in label or "synthetic" in label:
            ai_score = max(ai_score or 0.0, score)
        elif "real" in label or "authentic" in label:
            real_score = max(real_score or 0.0, score)

    if ai_score is not None:
        return ai_score
    if real_score is not None:
        return 1.0 - real_score

    top = max(outputs, key=lambda item: float(item.get("score", 0.0)))
    label = str(top.get("label", "")).lower()
    score = float(top.get("score", 0.0))
    if "real" in label or "authentic" in label:
        return 1.0 - score
    return score


@lru_cache(maxsize=1)
def _get_convnext():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("PyTorch is required for model inference") from exc

    try:
        import timm  # type: ignore
        from timm.data import create_transform, resolve_data_config  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("timm is required for ConvNeXt inference") from exc

    checkpoint_path = _checkpoint_path()
    if not checkpoint_path.exists():
        raise RuntimeError("MODEL_CHECKPOINT not found. Set MODEL_CHECKPOINT to a valid checkpoint path.")

    model_name = _convnext_model_name()
    model = timm.create_model(model_name, pretrained=False, num_classes=2)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state: dict[str, Any] | None = None
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    if state is None:
        if isinstance(checkpoint, dict):
            state = checkpoint  # type: ignore[assignment]
        else:
            raise RuntimeError("Unsupported checkpoint format")

    cleaned = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    device = _resolve_torch_device()
    model.to(device)
    model.eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform, device


def probability_from_bytes(payload: bytes) -> float:
    if not payload:
        return 0.0

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("Pillow is required to read images") from exc

    image = Image.open(io.BytesIO(payload)).convert("RGB")
    backend = _backend()
    if backend in {"convnext", "local"}:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("PyTorch is required for model inference") from exc

        model, transform, device = _get_convnext()
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
        if probs.shape[-1] < 2:
            raise RuntimeError("ConvNeXt model returned unexpected output shape")
        return float(probs[0, 1].item())

    detector = _get_pipeline()
    outputs = detector(image)
    if not isinstance(outputs, list) or not outputs:
        raise RuntimeError("Model returned empty result")
    return _ai_probability_from_output(outputs)


@dataclass(frozen=True)
class VisualFeatureStats:
    width: int
    height: int
    brightness: float
    contrast: float
    saturation: float
    edge_density: float
    edge_strength: float
    channel_balance: float
    bytes_per_megapixel: float


def _visual_feature_stats_from_bytes(payload: bytes) -> VisualFeatureStats:
    try:
        from PIL import Image, ImageFilter, ImageStat  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("Pillow is required to read images") from exc

    image = Image.open(io.BytesIO(payload)).convert("RGB")
    width, height = image.size
    megapixels = max((width * height) / 1_000_000, 0.001)

    gray = image.convert("L")
    gray_stat = ImageStat.Stat(gray)
    hsv_stat = ImageStat.Stat(image.convert("HSV"))
    rgb_stat = ImageStat.Stat(image)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)

    channel_means = rgb_stat.mean
    return VisualFeatureStats(
        width=width,
        height=height,
        brightness=float(gray_stat.mean[0]),
        contrast=float(gray_stat.stddev[0]),
        saturation=float(hsv_stat.mean[1]),
        edge_density=float(edge_stat.mean[0]) / 255.0,
        edge_strength=float(edge_stat.stddev[0]) / 255.0,
        channel_balance=(max(channel_means) - min(channel_means)) / 255.0,
        bytes_per_megapixel=len(payload) / megapixels,
    )


def _average_visual_feature_stats(items: list[VisualFeatureStats]) -> VisualFeatureStats:
    count = len(items)
    if count == 1:
        return items[0]

    return VisualFeatureStats(
        width=round(sum(item.width for item in items) / count),
        height=round(sum(item.height for item in items) / count),
        brightness=sum(item.brightness for item in items) / count,
        contrast=sum(item.contrast for item in items) / count,
        saturation=sum(item.saturation for item in items) / count,
        edge_density=sum(item.edge_density for item in items) / count,
        edge_strength=sum(item.edge_strength for item in items) / count,
        channel_balance=sum(item.channel_balance for item in items) / count,
        bytes_per_megapixel=sum(item.bytes_per_megapixel for item in items) / count,
    )


def _visual_signs_from_stats(stats: VisualFeatureStats) -> list[str]:
    signs: list[str] = []

    if stats.contrast < 35:
        signs.append("низкий локальный контраст: детали выглядят сглаженными")
    elif stats.contrast > 75:
        signs.append("очень высокий контраст: возможна сильная обработка или генеративная стилизация")
    else:
        signs.append("контраст в обычном диапазоне")

    if stats.edge_density < 0.025:
        signs.append("мало выраженных границ: текстуры и мелкие детали могут быть замылены")
    elif stats.edge_density > 0.12:
        signs.append("много резких границ: заметны плотные контуры или артефакты детализации")
    else:
        signs.append("плотность границ выглядит умеренной")

    if stats.edge_strength < 0.055:
        signs.append("низкая резкость: изображение похоже на сглаженное или сжатое")
    elif stats.edge_strength > 0.18:
        signs.append("повышенная резкость: возможны следы шарпинга или искусственной детализации")

    if stats.saturation < 45:
        signs.append("низкая насыщенность: цвета выглядят приглушенными")
    elif stats.saturation > 115:
        signs.append("высокая насыщенность: цвета выглядят усиленными")

    if stats.channel_balance < 0.035:
        signs.append("каналы RGB слишком ровные: цветовой баланс выглядит необычно нейтральным")
    elif stats.channel_balance > 0.32:
        signs.append("сильный перекос цветовых каналов: возможна заметная цветокоррекция")

    if stats.bytes_per_megapixel < 90_000:
        signs.append("низкая плотность данных: файл сильно сжат относительно разрешения")
    elif stats.bytes_per_megapixel > 1_800_000:
        signs.append("высокая плотность данных: файл содержит много деталей или слабое сжатие")

    return signs[:6]


def visual_signs_from_bytes(payload: bytes) -> list[str]:
    if not payload:
        return ["файл пустой, признаки извлечь нельзя"]
    return _visual_signs_from_stats(_visual_feature_stats_from_bytes(payload))


def visual_signs_from_frames(frames: list[bytes]) -> list[str]:
    if not frames:
        return ["кадры не извлечены, признаки видео оценить нельзя"]

    stats = [_visual_feature_stats_from_bytes(frame) for frame in frames]
    signs = _visual_signs_from_stats(_average_visual_feature_stats(stats))
    signs.insert(0, f"признаки усреднены по кадрам: {len(frames)}")
    return signs[:7]


def label_from_probability(probability: float) -> str:
    if probability >= 0.75:
        return "Высокая вероятность AI-генерации"
    if probability >= 0.5:
        return "Средняя вероятность AI-генерации"
    if probability >= 0.25:
        return "Низкая вероятность AI-генерации"
    return "Контент, вероятно, оригинальный"


def explanation_from_probability(probability: float) -> str:
    if probability >= 0.75:
        return (
            "Модель обнаружила сильный набор статистических признаков синтетики "
            "(неестественные текстуры, артефакты деталей и частотные паттерны), "
            "характерных для AI-генерации."
        )
    if probability >= 0.5:
        return (
            "Модель видит заметную долю признаков, типичных для AI-контента, "
            "но результат остается вероятностным и требует осторожной интерпретации."
        )
    if probability >= 0.25:
        return (
            "Обнаружены отдельные слабые признаки синтетики, "
            "но их недостаточно для уверенного вывода о генерации ИИ."
        )
    return (
        "Выраженных признаков AI-генерации почти нет: "
        "паттерны ближе к оригинальному контенту."
    )



