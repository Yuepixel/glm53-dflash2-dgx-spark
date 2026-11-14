#!/usr/bin/env python3
"""check_links.py - fail if a relative Markdown link points at a missing file.

Only relative links are checked (no network). Anchors and external URLs are skipped.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r'\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
SKIP_DIRS = {".git", "__pycache__", "node_modules"}

bad = []
checked = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for name in filenames:
        if not name.endswith(".md"):
            continue
        path = os.path.join(dirpath, name)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for match in LINK.finditer(text):
            target = match.group(1)
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
                continue
            if target.startswith(("#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            checked += 1
            resolved = os.path.normpath(os.path.join(dirpath, target))
            if not os.path.exists(resolved):
                bad.append("%s -> %s" % (os.path.relpath(path, ROOT), target))

if bad:
    print("broken relative links (%d):" % len(bad))
    for item in bad:
        print("  " + item)
    sys.exit(1)
print("relative links ok (%d checked)" % checked)
