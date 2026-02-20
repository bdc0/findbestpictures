import subprocess
import json
import re
import argparse
import sys
import os
from datetime import datetime, timedelta

# Try to import OpenCV
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

def parse_ls_l(directory="."):
    """
    Runs 'ls -lT' on the specified directory and parses the output 
    into a list of dictionaries.
    """
    try:
        # Run ls -lT command (macOS specific for full time info)
        result = subprocess.run(['ls', '-lT', directory], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        parsed_files = []
        
        # Split output into lines
        lines = output.split('\n')
        
        # Regex to parse ls -lT output
        # Columns: perms, links, owner, group, size, Month, Day, H:M:S, Year, filename
        # Example: -rw-r--r--  1 user  group  123 Feb 19 12:00:00 2026 filename.txt
        ls_pattern = re.compile(r'^([drwxstlbcps-]{10}[@+]?)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\w+\s+\d+\s+\d{2}:\d{2}:\d{2}\s+\d{4})\s+(.+)$')

        for line in lines:
            # Skip total line
            if line.startswith('total'):
                continue
            
            # Skip directories (permissions start with 'd')
            if line.startswith('d'):
                continue
                
            match = ls_pattern.match(line)
            if match:
                date_str = match.group(6)
                # Parse date: Feb 19 12:00:00 2026
                dt_object = datetime.strptime(date_str, "%b %d %H:%M:%S %Y")
                
                filename = match.group(7)
                # Handle full path if directory is specified
                full_path = os.path.join(directory, filename)
                
                file_info = {
                    'permissions': match.group(1),
                    'links': int(match.group(2)),
                    'owner': match.group(3),
                    'group': match.group(4),
                    'size': int(match.group(5)),
                    'date': date_str,
                    'datetime': dt_object.isoformat(), # Store as ISO for JSON
                    'timestamp': dt_object.timestamp(), # Store timestamp for sorting/calc
                    'name': filename,
                    'path': full_path,
                    'original_line': line
                }
                parsed_files.append(file_info)
            else:
                pass
                
        return parsed_files

    except subprocess.CalledProcessError as e:
        print(f"Error running ls: {e}", file=sys.stderr)
        return []

def group_files_by_time(files, delta_seconds=30):
    """
    Groups files where each file is created less than 'delta_seconds' 
    from the previous one.
    """
    if not files:
        return []
        
    # Sort by timestamp
    files.sort(key=lambda x: x['timestamp'])
    
    groups = []
    current_group = [files[0]]
    
    for i in range(1, len(files)):
        prev_file = files[i-1]
        curr_file = files[i]
        
        time_diff = curr_file['timestamp'] - prev_file['timestamp']
        
        if time_diff < delta_seconds:
            current_group.append(curr_file)
        else:
            groups.append(current_group)
            current_group = [curr_file]
            
    if current_group:
        groups.append(current_group)
        
    return groups

def is_image_file(filename):
    """Checks if a file is an image based on extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.heic', '.heif'}
    return os.path.splitext(filename)[1].lower() in image_extensions

def get_orb_descriptor(image_path):
    """Computes ORB descriptor for an image.
    
    For HEIC/HEIF files that OpenCV can't read, automatically tries the
    converted JPG in the 'jpg' subdirectory (created by --convert-heic).
    """
    if not HAS_OPENCV:
        return None
    
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try converted JPG for HEIC/HEIF files
            ext = os.path.splitext(image_path)[1].lower()
            if ext in ('.heic', '.heif'):
                parent_dir = os.path.dirname(image_path)
                basename = os.path.splitext(os.path.basename(image_path))[0]
                jpg_path = os.path.join(parent_dir, "jpg", basename + ".jpg")
                if os.path.exists(jpg_path):
                    img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image '{image_path}'. Skipping visual check.", file=sys.stderr)
                return None
        
        orb = cv2.ORB_create()
        _, des = orb.detectAndCompute(img, None)
        return des
    except Exception as e:
        print(f"Error processing image {image_path}: {e}", file=sys.stderr)
        return None

def get_phash(image_path):
    """Computes a 64-bit dHash (difference hash) for an image.
    
    Resizes the image to 9x8, converts to grayscale, and compares 
    adjacent pixels to create the hash.
    """
    if not HAS_OPENCV:
        return None
        
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try converted JPG for HEIC/HEIF files
            ext = os.path.splitext(image_path)[1].lower()
            if ext in ('.heic', '.heif'):
                parent_dir = os.path.dirname(image_path)
                basename = os.path.splitext(os.path.basename(image_path))[0]
                jpg_path = os.path.join(parent_dir, "jpg", basename + ".jpg")
                if os.path.exists(jpg_path):
                    img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # We already warn in get_orb_descriptor if both fail, 
                # but since this might be called independently, let's be safe.
                # However, usually filter_similar_images calls one or the other.
                return None

        # Resize to 9x8 for dHash
        resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        
        # Compute difference between adjacent pixels in same row
        # (result is 8 bits per row * 8 rows = 64 bits)
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert boolean array to integer hash
        return sum([2**i for i, v in enumerate(diff.flatten()) if v])
    except Exception:
        return None

def hamming_distance(h1, h2):
    """Calculates Hamming distance between two 64-bit hashes."""
    if h1 is None or h2 is None:
        return 999
    # Count set bits in XOR of two hashes
    return bin(h1 ^ h2).count('1')

def are_images_similar(des1, des2, threshold=0.75):
    """
    Compares two ORB descriptors using BFMatcher.
    Returns True if they are similar.
    """
    if des1 is None or des2 is None:
        return False
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if not matches:
        return False
        
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate similarity score based on number of good matches relative to keypoints
    # This is a heuristic. 
    # Logic: If a significant portion of keypoints match well, they are similar.
    
    # Simple check: if we have "enough" matches. 
    # For ORB (default 500 keypoints), let's say 20-30 good matches usually implies similarity 
    # for near-duplicates.
    # However, let's use a ratio.
    
    # Let's say if > 20% of the keypoints match, it's similar.
    num_keypoints = min(len(des1), len(des2))
    if num_keypoints == 0:
        return False
        
    match_ratio = len(matches) / num_keypoints
    
    # Using a simpler heuristic for now: Top 30 matches having low distance?
    # Or just total match count > 50?
    
    # Let's use a match count threshold for robustness + distance check.
    # Top 10 matches average distance < 50?
    
    # Let's stick to the prompt: "visually similar". 
    # ORB isn't perfect for "semantic" similarity, but good for "duplicate/near-duplicate".
    
    # Standard approach:
    good_matches = [m for m in matches if m.distance < 50]
    
    # If we have > threshold good matches, call it similar for now.
    return len(good_matches) > threshold


def filter_similar_images(group, method='orb', threshold=None):
    """
    Filters a group of files. For visually similar images, keeps only the first one.
    Retains all non-image files.
    """
    if not HAS_OPENCV:
        return group
        
    if threshold is None:
        threshold = 10 if method == 'orb' else 10
        
    filtered_group = []
    # List of (file_info, features)
    kept_images = [] 
    
    for file_info in group:
        path = file_info['path']
        
        if not is_image_file(file_info['name']):
            filtered_group.append(file_info)
            continue
            
        # Extract features based on method
        if method == 'orb':
            features = get_orb_descriptor(path)
        else: # phash
            features = get_phash(path)
            
        is_similar = False
        for kept_file, kept_features in kept_images:
            if method == 'orb':
                if are_images_similar(features, kept_features, threshold=threshold):
                    is_similar = True
                    break
            else: # phash
                if hamming_distance(features, kept_features) <= threshold:
                    is_similar = True
                    break
        
        if not is_similar:
            filtered_group.append(file_info)
            if features is not None:
                kept_images.append((file_info, features))
    
    return filtered_group

import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the best pictures by grouping by timestamp and filtering visual duplicates."
    )
    parser.add_argument("directory", nargs="?", default=".", help="Directory containing images to process (default: current directory)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output (useful with --copy or --convert-heic)")
    parser.add_argument("--copy", action="store_true", help="Copy filtered unique files to a 'unique' subdirectory")
    parser.add_argument("--method", choices=['orb', 'phash'], default='orb', help="Visual similarity method (default: orb)")
    parser.add_argument("--threshold", type=int, help="Similarity threshold (ORB: match count, pHash: hamming distance)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print processing statistics to stderr")
    parser.add_argument("--convert-heic", action="store_true", help="Convert HEIC images to JPG (in a 'jpg' subdir) before processing")
    args = parser.parse_args()

    # Ensure valid directory for copy operation
    target_dir = args.directory
    if args.copy:
        unique_dir = os.path.join(target_dir, "unique")
        if not os.path.exists(unique_dir):
            try:
                os.makedirs(unique_dir)
            except OSError as e:
                print(f"Error creating directory {unique_dir}: {e}", file=sys.stderr)
                sys.exit(1)
    # Convert HEIC to JPG if requested
    if args.convert_heic:
        jpg_dir = os.path.join(target_dir, "jpg")
        if os.path.exists(jpg_dir):
            print(f"Error: '{jpg_dir}' already exists. Remove it first.", file=sys.stderr)
            sys.exit(1)
        os.makedirs(jpg_dir)
        if args.verbose:
            print(f"Converting HEIC files to JPG in '{jpg_dir}'...", file=sys.stderr)
        try:
            result = subprocess.run(
                ['magick', 'mogrify', '-path', jpg_dir, '-format', 'jpg', '*.HEIC'],
                cwd=target_dir,
                capture_output=True, text=True,
            )
            # Also try lowercase .heic
            subprocess.run(
                ['magick', 'mogrify', '-path', jpg_dir, '-format', 'jpg', '*.heic'],
                cwd=target_dir,
                capture_output=True, text=True,
            )
            if args.verbose:
                # Count converted files
                jpg_count = len([f for f in os.listdir(jpg_dir) if f.lower().endswith('.jpg')])
                print(f"Converted: {jpg_count} JPG files", file=sys.stderr)
        except FileNotFoundError:
            print("Error: 'magick' command not found. Install ImageMagick.", file=sys.stderr)
            sys.exit(1)

    files = parse_ls_l(args.directory)
    if args.verbose:
        print(f"Files found: {len(files)}", file=sys.stderr)

    groups = group_files_by_time(files)
    if args.verbose:
        group_sizes = [len(g) for g in groups]
        print(f"Groups: {len(groups)} ({', '.join(str(s) for s in group_sizes)})", file=sys.stderr)
    
    # Filter groups for similarity
    if HAS_OPENCV:
        # Default thresholds
        if args.threshold is None:
            threshold = 10 # works for both methods by default coincidentally
        else:
            threshold = args.threshold
            
        filtered_groups = [filter_similar_images(g, method=args.method, threshold=threshold) for g in groups]
        groups = filtered_groups
    elif not args.quiet:
         print("Warning: OpenCV not found. Visual similarity check skipped.", file=sys.stderr)
    
    # Filter out empty groups (in case filtering removed all items or logic produced empties)
    groups = [g for g in groups if g]

    unique_count = sum(len(g) for g in groups)
    if args.verbose:
        print(f"Unique files: {unique_count}", file=sys.stderr)

    # Copy files if requested
    if args.copy:
        count = 0
        for group in groups:
            for file_info in group:
                try:
                    src = file_info['path']
                    # Keep filename, copy to unique_dir
                    dst = os.path.join(unique_dir, file_info['name'])
                    # copy2 preserves metadata (timestamps) which might be useful
                    shutil.copy2(src, dst)
                    count += 1
                except Exception as e:
                    print(f"Error copying {file_info['name']}: {e}", file=sys.stderr)
        
        if not args.quiet:
            print(f"Copied {count} files to '{unique_dir}'")

    if args.quiet:
        sys.exit(0)

    # Print groups separated by blank line
    first = True
    for group in groups:
        if not first:
            print() # Blank line between sets
        first = False
        
        if args.json:
            print(json.dumps(group, indent=2))
        else:
            for file in group:
                print(file['original_line'])
