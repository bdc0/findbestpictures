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

def calculate_focus_score(image_path):
    """
    Calculates the focus score of an image using the Laplacian variance method.
    Higher score indicates a sharper image.
    """
    if not HAS_OPENCV:
        return 0.0
        
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
                return 0.0
        
        # Calculate Laplacian variance
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except Exception:
        return 0.0

def hamming_distance(h1, h2):
    """Calculates Hamming distance between two 64-bit hashes."""
    if h1 is None or h2 is None:
        return 999
    # Count set bits in XOR of two hashes
    return bin(h1 ^ h2).count('1')

def are_images_similar(des1, des2, threshold=10):
    """
    Compares two ORB descriptors using BFMatcher.
    Returns (is_similar, match_count).
    """
    if des1 is None or des2 is None:
        return False, 0
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if not matches:
        return False, 0
        
    # Standard approach: count matches with distance < 50
    good_matches = [m for m in matches if m.distance < 50]
    
    return len(good_matches) > threshold, len(good_matches)


def filter_similar_images(group, method='orb', threshold=None, use_focus=True):
    """
    Filters a group of files. For visually similar images, keeps only the first one
    (or the sharpest one if use_focus is True).
    Retains all non-image files.
    
    Updates file_info with similarity metadata.
    """
    if not HAS_OPENCV:
        return group
        
    if threshold is None:
        threshold = 10 if method == 'orb' else 10
        
    # List of (file_info, features, focus_score)
    kept_images = [] 
    
    for file_info in group:
        path = file_info['path']
        file_info['is_duplicate'] = False
        file_info['similarity_to'] = None
        file_info['similarity_score'] = None
        file_info['focus_score'] = 0.0
        
        if not is_image_file(file_info['name']):
            continue
            
        # Get focus score if enabled
        focus_score = 0.0
        if use_focus:
            focus_score = calculate_focus_score(path)
        file_info['focus_score'] = focus_score

        # Extract features based on method
        if method == 'orb':
            features = get_orb_descriptor(path)
        else: # phash
            features = get_phash(path)
            
        match_idx = -1
        last_score = None
        
        for idx, (kept_file, kept_features, kept_focus) in enumerate(kept_images):
            if method == 'orb':
                similar, score = are_images_similar(features, kept_features, threshold=threshold)
                if similar:
                    match_idx = idx
                    last_score = score
                    break
            else: # phash
                dist = hamming_distance(features, kept_features)
                if dist <= threshold:
                    match_idx = idx
                    last_score = dist
                    break
        
        if match_idx == -1:
            # New unique image (so far)
            if features is not None:
                kept_images.append((file_info, features, focus_score))
        else:
            # Found a match. Compare focus scores to see which to keep.
            kept_file, _, kept_focus = kept_images[match_idx]
            
            if use_focus and focus_score > kept_focus:
                # This image is sharper! Swap it in as the representative.
                # Mark the old one as duplicate of this one
                kept_file['is_duplicate'] = True
                kept_file['similarity_to'] = file_info['name']
                # Re-using the same match score since they are similar
                kept_file['similarity_score'] = last_score
                
                # Replace representative in kept_images
                kept_images[match_idx] = (file_info, features, focus_score)
            else:
                # Kept one is sharper (or focus is disabled). Mark this one as duplicate.
                file_info['is_duplicate'] = True
                file_info['similarity_to'] = kept_file['name']
                file_info['similarity_score'] = last_score
    
    return group

def generate_html_report(groups, output_path, target_dir):
    """Generates a premium HTML gallery report of the image groups."""
    doc_title = f"Image Selection Report - {os.path.basename(target_dir)}"
    
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"    <title>{doc_title}</title>",
        "    <style>",
        "        :root {",
        "            --bg: #0f172a;",
        "            --card-bg: #1e293b;",
        "            --text: #f8fafc;",
        "            --text-muted: #94a3b8;",
        "            --accent: #38bdf8;",
        "            --success: #22c55e;",
        "            --border: #334155;",
        "        }",
        "        body {",
        "            background: var(--bg);",
        "            color: var(--text);",
        "            font-family: 'Inter', system-ui, -apple-system, sans-serif;",
        "            margin: 0;",
        "            padding: 2rem;",
        "            line-height: 1.5;",
        "        }",
        "        .container { max-width: 1400px; margin: 0 auto; }",
        "        header { margin-bottom: 3rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }",
        "        h1 { font-size: 1.875rem; font-weight: 700; margin: 0; color: var(--accent); }",
        "        .stats { color: var(--text-muted); font-size: 0.875rem; margin-top: 0.5rem; }",
        "        .group { border: 1px solid var(--border); border-radius: 1rem; padding: 1.5rem; margin-bottom: 3rem; background: rgba(30, 41, 59, 0.5); }",
        "        .group-header { margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem; }",
        "        .group-title { font-size: 1.125rem; font-weight: 600; color: var(--text-muted); }",
        "        .row { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.5rem; }",
        "        .card { position: relative; border-radius: 0.75rem; overflow: hidden; background: var(--card-bg); border: 1px solid var(--border); transition: transform 0.2s, box-shadow 0.2s; }",
        "        .card:hover { transform: translateY(-4px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4); }",
        "        .card.keep { border-color: var(--success); box-shadow: 0 0 15px rgba(34, 197, 94, 0.2); }",
        "        .img-container { aspect-ratio: 4/3; background: #000; overflow: hidden; display: flex; align-items: center; justify-content: center; }",
        "        img { width: 100%; height: 100%; object-fit: cover; }",
        "        .badge { position: absolute; top: 0.75rem; left: 0.75rem; padding: 0.25rem 0.75rem; border-radius: 2rem; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }",
        "        .badge-keep { background: var(--success); color: #fff; }",
        "        .badge-dup { background: rgba(0, 0, 0, 0.6); color: var(--text-muted); backdrop-filter: blur(4px); }",
        "        .card-info { padding: 1rem; }",
        "        .filename { font-size: 0.875rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 0.25rem; }",
        "        .meta { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted); }",
        "        .highlight { color: var(--accent); font-weight: 600; }",
        "        @media (max-width: 640px) { body { padding: 1rem; } .row { grid-template-columns: 1fr; } }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        "        <header>",
        f"            <h1>{doc_title}</h1>",
        f"            <div class='stats'>Found {len(groups)} groups with {sum(len(g) for g in groups)} total files</div>",
        "        </header>",
        "        <main>"
    ]
    
    for i, group in enumerate(groups, 1):
        html.append(f"            <section class='group'>")
        html.append(f"                <div class='group-header'><span class='group-title'>Set {i}</span></div>")
        html.append(f"                <div class='row'>")
        
        for f in group:
            is_dup = f.get('is_duplicate', False)
            keep_class = "keep" if not is_dup else ""
            badge = "KEEP" if not is_dup else "DUP"
            badge_class = "badge-keep" if not is_dup else "badge-dup"
            
            focus_score = f.get('focus_score', 0)
            focus_str = f"{focus_score:.1f}" if focus_score else "N/A"
            
            # Use absolute path for local viewing
            original_path = f['path']
            img_src = f"file://{original_path}"
            
            # Browser HEIC Fallback: Check for converted JPG
            ext = os.path.splitext(f['name'])[1].lower()
            if ext in ('.heic', '.heif'):
                parent_dir = os.path.dirname(original_path)
                basename = os.path.splitext(f['name'])[0]
                jpg_path = os.path.join(parent_dir, "jpg", basename + ".jpg")
                if os.path.exists(jpg_path):
                    img_src = f"file://{jpg_path}"
                else:
                    # No JPG fallback? The browser likely won't display the HEIC
                    pass
            
            html.append(f"                    <div class='card {keep_class}'>")
            html.append(f"                        <span class='badge {badge_class}'>{badge}</span>")
            html.append(f"                        <div class='img-container'><img src='{img_src}' alt='{f['name']}' loading='lazy'></div>")
            html.append(f"                        <div class='card-info'>")
            html.append(f"                            <div class='filename' title='{f['name']}'>{f['name']}</div>")
            html.append(f"                            <div class='meta'>")
            html.append(f"                                <span>Focus: <span class='highlight'>{focus_str}</span></span>")
            if is_dup:
                html.append(f"                                <span>Match: {f.get('similarity_to', 'N/A')}</span>")
            html.append(f"                            </div>")
            html.append(f"                        </div>")
            html.append(f"                    </div>")
            
        html.append(f"                </div>")
        html.append(f"            </section>")
        
    html.append("        </main>")
    html.append("    </div>")
    html.append("</body>")
    html.append("</html>")
    
    try:
        with open(output_path, 'w') as f:
            f.write("\n".join(html))
        return True
    except Exception as e:
        print(f"Error writing HTML report: {e}", file=sys.stderr)
        return False

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
    parser.add_argument("--clean", action="store_true", help="Remove the 'jpg' directory after processing (use only with --convert-heic)")
    parser.add_argument("--no-focus", action="store_false", dest="focus", default=True, help="Disable focus-based selection (keep the first similar image instead)")
    parser.add_argument("--html", help="Generate an HTML gallery report to the specified file")
    args = parser.parse_args()

    # Validation
    if args.clean and not args.convert_heic:
        print("Error: --clean can only be used with --convert-heic.", file=sys.stderr)
        sys.exit(1)

    # Ensure valid directory for copy operation
    target_dir = os.path.abspath(args.directory)
    if args.verbose:
        print(f"Target directory: {target_dir}", file=sys.stderr)

    if args.copy:
        unique_dir = os.path.join(target_dir, "unique")
        if args.verbose:
            print(f"Unique output directory: {unique_dir}", file=sys.stderr)
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

    # Parse and group files
    files = parse_ls_l(target_dir)
    if args.verbose:
        print(f"Files found: {len(files)}", file=sys.stderr)

    groups = group_files_by_time(files)
    
    # Filter groups for similarity
    if HAS_OPENCV:
        # Default thresholds
        threshold = args.threshold if args.threshold is not None else 10
        groups = [filter_similar_images(g, method=args.method, threshold=threshold, use_focus=args.focus) for g in groups]
    elif not args.quiet:
         print("Warning: OpenCV not found. Visual similarity check skipped.", file=sys.stderr)
    
    # Filter out empty groups 
    groups = [g for g in groups if g]

    # Generate HTML report if requested
    if args.html:
        if args.verbose:
            print(f"Generating HTML report: {args.html}...", file=sys.stderr)
        generate_html_report(groups, args.html, target_dir)

    # Calculate final unique count
    unique_count = sum(1 for group in groups for f in group if not f.get('is_duplicate'))

    # Output detailed stats to stderr if verbose
    if args.verbose:
        print(f"Groups: {len(groups)}", file=sys.stderr)
        print("\n" + "="*50, file=sys.stderr)
        print("SIMILARITY BREAKDOWN BY GROUP", file=sys.stderr)
        print("="*50, file=sys.stderr)
        
        for i, group in enumerate(groups, 1):
            print(f"\nGroup {i} ({len(group)} files):", file=sys.stderr)
            for f in group:
                focus_info = f" (focus: {f.get('focus_score', 0):.1f})" if f.get('focus_score') else ""
                if f.get('is_duplicate'):
                    method_label = "dist" if args.method == 'phash' else "matches"
                    print(f"  [DUP]  {f['name']:30} (matched {f['similarity_to']} with {f['similarity_score']} {method_label}){focus_info}", file=sys.stderr)
                else:
                    print(f"  [KEEP] {f['name']:30}{focus_info}", file=sys.stderr)
        
        print("\n" + "="*50, file=sys.stderr)
        print(f"FINISH: Unique files: {unique_count}", file=sys.stderr)
        print("="*50 + "\n", file=sys.stderr)

    # Perform file operations
    if args.copy:
        count = 0
        for group in groups:
            for f in group:
                if not f.get('is_duplicate'):
                    try:
                        shutil.copy2(f['path'], os.path.join(unique_dir, f['name']))
                        count += 1
                    except Exception as e:
                        print(f"Error copying {f['name']}: {e}", file=sys.stderr)
        if not args.quiet:
            print(f"Copied {count} files to '{unique_dir}'")

    # Output to stdout
    if not args.quiet:
        if args.json:
            result_files = [f for group in groups for f in group if not f.get('is_duplicate')]
            print(json.dumps(result_files, indent=2))
        else:
            first = True
            for group in groups:
                if not first:
                    print()
                first = False
                for f in group:
                    if not f.get('is_duplicate'):
                        print(f['original_line'])

    # Cleanup HEIC temporary files
    if args.convert_heic and args.clean:
        jpg_dir = os.path.join(target_dir, "jpg")
        if os.path.exists(jpg_dir):
            if args.verbose:
                print(f"Cleaning up temporary directory '{jpg_dir}'...", file=sys.stderr)
            try:
                shutil.rmtree(jpg_dir)
            except Exception as e:
                print(f"Error cleaning up '{jpg_dir}': {e}", file=sys.stderr)

