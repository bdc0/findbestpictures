#!/usr/bin/env python3
"""
Creates test files for verifying ls_parser.py functionality.

Generates two test directories:
  test_grouping/  — Files with controlled timestamps for time-grouping tests.
  test_images/    — PNG images for visual-similarity tests.

Usage:
    python3 create_test_files.py          # Create both test sets
    python3 create_test_files.py --clean  # Remove test directories
"""

import os
import sys
import shutil
import argparse
import subprocess
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GROUPING_DIR = os.path.join(SCRIPT_DIR, "test_grouping")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "test_images")


# ---------------------------------------------------------------------------
# Time-grouping test files
# ---------------------------------------------------------------------------
def create_grouping_tests():
    """
    Creates files with specific timestamps to exercise group_files_by_time().

    Layout (default 30s delta):
        Group 1:  file1 (T+0s), file2 (T+10s)        — within 30s of each other
        Group 2:  file3 (T+50s), file4 (T+60s)        — >30s gap from group 1
        Singleton: file5 (T+200s)                      — isolated file
    """
    os.makedirs(GROUPING_DIR, exist_ok=True)

    base = datetime(2026, 2, 19, 12, 0, 0)
    files = {
        "file1.txt": base,                                # Group 1
        "file2.txt": base + timedelta(seconds=10),        # Group 1
        "file3.txt": base + timedelta(seconds=50),        # Group 2
        "file4.txt": base + timedelta(seconds=60),        # Group 2
        "file5.txt": base + timedelta(seconds=200),       # Singleton
    }

    for name, ts in files.items():
        path = os.path.join(GROUPING_DIR, name)
        # Create the file
        with open(path, "w") as f:
            f.write(f"Test file created at {ts.isoformat()}\n")

        # Set modification time via `touch -t` (macOS format: [[CC]YY]MMDDhhmm[.SS])
        stamp = ts.strftime("%Y%m%d%H%M.%S")
        subprocess.run(["touch", "-t", stamp, path], check=True)

    print(f"✓ Created {len(files)} grouping test files in {GROUPING_DIR}/")
    print("  Expected groups (30s delta):")
    print("    Group 1: file1.txt, file2.txt")
    print("    Group 2: file3.txt, file4.txt")
    print("    Group 3: file5.txt")


# ---------------------------------------------------------------------------
# Image similarity test files
# ---------------------------------------------------------------------------
def create_image_tests():
    """
    Creates PNG images to exercise the visual-similarity filter.

    Images:
        img1.png — Base image (random colour blocks)
        img2.png — Exact duplicate of img1
        img3.png — Slightly modified img1 (a few pixels changed)
        img4.png — Completely different image

    All are timestamped within 30s so they land in the same group.
    Expected result: img2 and img3 are filtered as similar to img1.
                     img1 and img4 survive.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("⚠  Skipping image tests — opencv-python is not installed.", file=sys.stderr)
        print("   Install with: pip install opencv-python", file=sys.stderr)
        return

    os.makedirs(IMAGES_DIR, exist_ok=True)

    # --- img1: base image with random blocks giving ORB features to detect ---
    np.random.seed(42)
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw random rectangles for texture
    for _ in range(40):
        x1, y1 = np.random.randint(0, 250, 2)
        x2, y2 = x1 + np.random.randint(10, 50), y1 + np.random.randint(10, 50)
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.rectangle(img1, (x1, y1), (x2, y2), color, -1)

    # --- img2: exact copy ---
    img2 = img1.copy()

    # --- img3: near-duplicate (slight modification) ---
    img3 = img1.copy()
    img3[140:160, 140:160] = [0, 0, 255]  # small red patch

    # --- img4: completely different image (diagonal stripes + text) ---
    img4 = np.full((300, 300, 3), 230, dtype=np.uint8)  # light grey background
    # Diagonal stripes
    for i in range(-300, 600, 12):
        cv2.line(img4, (i, 0), (i + 300, 300), (20, 20, 120), 3)
    # Add text for additional distinct features
    cv2.putText(img4, "DIFFERENT", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    # Crosshatch overlay
    for i in range(-300, 600, 20):
        cv2.line(img4, (i + 300, 0), (i, 300), (120, 20, 20), 2)

    images = {
        "img1.png": img1,
        "img2.png": img2,
        "img3.png": img3,
        "img4.png": img4,
    }

    base = datetime(2026, 2, 19, 13, 0, 0)
    for i, (name, img) in enumerate(images.items()):
        path = os.path.join(IMAGES_DIR, name)
        cv2.imwrite(path, img)
        # Timestamp within 30s so they group together
        ts = base + timedelta(seconds=i * 5)
        stamp = ts.strftime("%Y%m%d%H%M.%S")
        subprocess.run(["touch", "-t", stamp, path], check=True)

    print(f"✓ Created {len(images)} test images in {IMAGES_DIR}/")
    print("  Expected similarity results:")
    print("    Kept:     img1.png, img4.png")
    print("    Filtered: img2.png (duplicate), img3.png (near-duplicate)")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def clean():
    """Remove test directories."""
    for d in [GROUPING_DIR, IMAGES_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"✓ Removed {d}/")
        else:
            print(f"  {d}/ does not exist, nothing to remove.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create (or clean) test files for ls_parser.py")
    parser.add_argument("--clean", action="store_true", help="Remove test directories instead of creating them")
    args = parser.parse_args()

    if args.clean:
        clean()
    else:
        create_grouping_tests()
        print()
        create_image_tests()
        print()
        print("Run ls_parser.py against the test dirs:")
        print(f"  python3 ls_parser.py {GROUPING_DIR}")
        print(f"  python3 ls_parser.py {IMAGES_DIR}")
