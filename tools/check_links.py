#!/usr/bin/env python3
"""Check that relative links and images in Markdown files point to real files.

External URLs and in-page anchors are not checked. Links inside fenced code
blocks and inline code are ignored.

Usage::

    python tools/check_links.py
"""

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent

MD_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_SRC = re.compile(r"<(?:img|a)\b[^>]*?(?:src|href)=\"([^\"]+)\"", re.IGNORECASE)
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")


def markdown_files():
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if (ROOT / line).exists()]


def links(path):
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = INLINE_CODE.sub("", line)
        for pattern in (MD_LINK, HTML_SRC):
            for match in pattern.finditer(line):
                yield lineno, match.group(1)


def is_external(target):
    return re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith(
        ("#", "//")
    )


def main():
    broken = []
    files = markdown_files()
    for path in files:
        for lineno, target in links(path):
            if is_external(target):
                continue
            file_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not file_part:
                continue
            resolved = (path.parent / file_part).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(ROOT)}:{lineno}: {target}")

    for line in broken:
        print(line)
    print(f"{len(broken)} broken relative link(s) in {len(files)} Markdown files")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
