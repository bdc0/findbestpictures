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
    """Computes ORB descriptor for an image."""
    if not HAS_OPENCV:
        return None
    
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # warn user
            print(f"Warning: Could not read image '{image_path}'. Skipping visual check.", file=sys.stderr)
            return None
        
        orb = cv2.ORB_create()
        _, des = orb.detectAndCompute(img, None)
        return des
    except Exception as e:
        print(f"Error processing image {image_path}: {e}", file=sys.stderr)
        return None

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


def filter_similar_images(group, threshold=10):
    """
    Filters a group of files. For visually similar images, keeps only the first one.
    Retains all non-image files.
    """
    if not HAS_OPENCV:
        # If OpenCV not available, return group as is (or warn?)
        return group
        
    filtered_group = []
    # List of (file_info, descriptor) for images kept so far in this group
    kept_images = [] 
    
    for file_info in group:
        path = file_info['path']
        
        if not is_image_file(file_info['name']):
            filtered_group.append(file_info)
            continue
            
        # It's an image
        des = get_orb_descriptor(path)
        
        is_similar = False
        for kept_file, kept_des in kept_images:
            if are_images_similar(des, kept_des, threshold=threshold):
                is_similar = True
                break
        
        if not is_similar:
            filtered_group.append(file_info)
            # Only store descriptor if we successfully computed one
            if des is not None:
                kept_images.append((file_info, des))
    
    return filtered_group

import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse 'ls -l' output and group by time.")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to parse (default: current directory)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format instead of original ls lines")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output")
    parser.add_argument("--copy", action="store_true", help="Copy unique files to a 'unique' subdirectory")
    parser.add_argument("--threshold", type=int, default=10, help="Minimum number of ORB matches for visual similarity (default: 10)")
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

    files = parse_ls_l(args.directory)
    groups = group_files_by_time(files)
    
    # Filter groups for similarity
    if HAS_OPENCV:
        filtered_groups = [filter_similar_images(g, threshold=args.threshold) for g in groups]
        groups = filtered_groups
    elif not args.quiet:
         print("Warning: OpenCV not found. Visual similarity check skipped.", file=sys.stderr)
    
    # Filter out empty groups (in case filtering removed all items or logic produced empties)
    groups = [g for g in groups if g]

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
