#!/usr/bin/env python3
"""Keep each note's "Related Scripts" section in sync with the script READMEs.

Every script README has a "Related Notes" section linking to notes. This tool
adds the reverse link to each of those notes: a "Related Scripts" section that
lists every script pointing at the note. Scripts already mentioned in the note's
section are kept; hand-written entries (a link followed by an explanation) are
left exactly as written, while bare links and legacy "Related Python Scripts"
tables are rewritten as a link plus the first sentence of the script's README.
The section is placed where the old one was, otherwise before "Exercises" (or
"References") at the same heading level.

Usage::

    python tools/sync_related_scripts.py          # update notes in place
    python tools/sync_related_scripts.py --check  # exit 1 if any note is out of date
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from generate_index import title_and_summary

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "notes"

SECTION_NAMES = ("related scripts", "related python scripts")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LINK = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]+)\s*\)")
SCRIPT_PATH = re.compile(r"scripts/(?:algorithms|plots|simulations)/[A-Za-z0-9_\-]+")
BARE_BULLET = re.compile(r"^\s*[-*]\s+\[[^\]]*\]\(\s*[^)\s]+\s*\)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


def headings(lines):
    """Yield (index, level, text) for headings outside fenced code blocks."""
    in_fence = False
    for i, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            match = HEADING.match(line)
            if match:
                yield i, len(match.group(1)), match.group(2)


def sections_named(lines, names):
    """Return [(start, end, level)] for every section whose title is in names."""
    heads = list(headings(lines))
    found = []
    for n, (i, level, text) in enumerate(heads):
        if text.strip().lower() in names:
            end = next((j for j, lvl, _ in heads[n + 1 :] if lvl <= level), len(lines))
            found.append((i, end, level))
    return found


def script_folders(base, lines):
    """Script folders referenced in lines by relative link or by a scripts/... path."""
    folders = set()
    for line in lines:
        candidates = [
            (base / target.split("#")[0]).resolve()
            for _, target in LINK.findall(line)
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target)
        ]
        candidates += [ROOT / path for path in SCRIPT_PATH.findall(line)]
        for target in candidates:
            folder = target if target.is_dir() else target.parent
            if (folder / "main.py").exists():
                folders.add(folder)
    return folders


def related_scripts_by_note():
    mapping = defaultdict(set)
    for readme in sorted((ROOT / "scripts").glob("*/*/README.md")):
        lines = readme.read_text(encoding="utf-8").splitlines()
        for start, end, _ in sections_named(lines, ("related notes",)):
            for _, target in LINK.findall("\n".join(lines[start + 1 : end])):
                path = (readme.parent / target.split("#")[0]).resolve()
                if path.suffix == ".md" and NOTES in path.parents and path.exists():
                    mapping[path].add(readme.parent)
    return mapping


def bullet(note, folder):
    title, summary = title_and_summary(folder / "README.md")
    summary = re.sub(r"^(?:This|The)\s+(?:Python\s+)?script\s+", "", summary)
    rel = os.path.relpath(folder, note.parent).replace(os.sep, "/")
    return f"- [{title}]({rel}/): {summary}" if summary else f"- [{title}]({rel}/)"


def updated_note(note, derived):
    original = note.read_text(encoding="utf-8")
    lines = original.splitlines()
    sections = sections_named(lines, SECTION_NAMES)

    folders, covered, preserved = set(derived), set(), []
    for start, end, _ in sections:
        body = lines[start + 1 : end]
        folders |= script_folders(note.parent, body)
        for line in body:
            if (
                not line.strip()
                or line.lstrip().startswith("|")
                or BARE_BULLET.match(line)
            ):
                continue
            preserved.append(line)
            covered |= script_folders(note.parent, [line])

    if not folders and not sections:
        return None

    generated = sorted(
        (bullet(note, folder) for folder in folders - covered), key=str.lower
    )
    body = preserved + generated

    if sections:
        insert_at, level = sections[0][0], sections[0][2]
        for start, end, _ in reversed(sections):
            del lines[start:end]
    else:
        anchor = sections_named(lines, ("exercises",)) or sections_named(
            lines, ("references",)
        )
        if anchor:
            insert_at, level = anchor[0][0], anchor[0][2]
        else:
            levels = [lvl for _, lvl, _ in list(headings(lines))[1:]]
            insert_at, level = len(lines), (min(levels) if levels else 2)

    block = [f"{'#' * level} Related Scripts", ""] + body + [""] if body else []
    if block and insert_at > 0 and lines[insert_at - 1].strip():
        block = [""] + block
    lines[insert_at:insert_at] = block

    text = "\n".join(lines).rstrip("\n") + "\n"
    return text if text != original else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if any note is stale"
    )
    args = parser.parse_args(argv)

    mapping = related_scripts_by_note()
    notes = sorted(p for p in NOTES.rglob("*.md") if p.name != "README.md")
    stale = []
    for note in notes:
        text = updated_note(note, mapping.get(note, set()))
        if text is None:
            continue
        stale.append(note.relative_to(ROOT))
        if not args.check:
            note.write_text(text, encoding="utf-8")

    verb = "out of date" if args.check else "updated"
    for path in stale:
        print(f"{verb}: {path}")
    print(f"{len(stale)} note(s) {verb}")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
