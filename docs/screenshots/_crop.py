"""Crop browser chrome + Windows taskbar off DataHub screenshots.

Chrome window at 1920x1080 native:
  - title bar / window controls: y 0..~30
  - tab bar:                     y ~30..60
  - address bar + bookmarks:     y ~60..~93
  DataHub content starts at ~y=95 (its own header row is next).

Windows taskbar: y ~1048..1080.

Uniform crop: (0, 95, 1920, 1050). Overwrites the PNGs in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

TOP = 118
BOTTOM = 1000
LEFT = 0
RIGHT = 1920

HERE = Path(__file__).resolve().parent


def main() -> int:
    shots = sorted(HERE.glob("[0-9][0-9]-*.png"))
    if not shots:
        print("no shots found", file=sys.stderr)
        return 1
    for p in shots:
        with Image.open(p) as im:
            if im.size != (1920, 1080):
                print(f"skip {p.name}: unexpected size {im.size}")
                continue
            cropped = im.crop((LEFT, TOP, RIGHT, BOTTOM))
            cropped.save(p, optimize=True)
        print(f"cropped {p.name} -> {cropped.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
