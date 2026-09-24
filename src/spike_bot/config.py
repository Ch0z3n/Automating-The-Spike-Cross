from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = ("capture", "court", "vision", "keys", "controller")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    return config


def save_config(path: Path, config: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

