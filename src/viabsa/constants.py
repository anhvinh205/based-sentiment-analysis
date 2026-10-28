"""Label space shared by every module (data, models, API, UI)."""

from __future__ import annotations

# Column order follows the raw CSV files.
ASPECTS: list[str] = [
    "stayingpower",
    "texture",
    "smell",
    "price",
    "others",
    "colour",
    "shipping",
    "packing",
]

# Per aspect the model predicts one of four classes.
LABELS: list[str] = ["none", "negative", "neutral", "positive"]
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = dict(enumerate(LABELS))
NONE_ID = LABEL2ID["none"]

N_ASPECTS = len(ASPECTS)
N_LABELS = len(LABELS)

ASPECT_DISPLAY: dict[str, dict[str, str]] = {
    "stayingpower": {"en": "Staying power", "vi": "Độ bền màu"},
    "texture": {"en": "Texture", "vi": "Chất son"},
    "smell": {"en": "Smell", "vi": "Mùi"},
    "price": {"en": "Price", "vi": "Giá"},
    "others": {"en": "Others", "vi": "Khác"},
    "colour": {"en": "Colour", "vi": "Màu sắc"},
    "shipping": {"en": "Shipping", "vi": "Giao hàng"},
    "packing": {"en": "Packing", "vi": "Đóng gói"},
}

TEXT_COLUMN = "text"
