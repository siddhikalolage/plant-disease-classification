import os
from pathlib import Path

from PIL import Image


def save_uploaded_file(uploaded_file, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return destination


def get_asset_path(filename: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    asset_path = root / "assets" / filename
    return asset_path if asset_path.exists() else None
