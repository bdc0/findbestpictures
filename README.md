# FindBestPictures

`ls_parser.py` is a tool designed to help you find the "best" pictures from a set of burst shots or similar images by grouping them by timestamp and filtering out visual duplicates.

## Features

- **Group by Time**: Automatically groups files that were taken within a short time window (default 30 seconds).
- **Visual Similarity Filter**: Uses ORB feature detection (via OpenCV) to identify and filter out near-identical images within each time group.
- **HEIC Support**: Converts Apple HEIC photos to JPG for similarity processing.
- **Multiple Output Formats**: Text (original `ls -l` lines) or structured JSON.
- **Workflow Friendly**: Supports copying unique files to a separate directory for easy review.

## Prerequisites

- **Python 3.x**
- **OpenCV** (Optional, required for visual similarity): `pip install opencv-python numpy`
- **ImageMagick** (Optional, required for HEIC conversion): `brew install imagemagick` (on macOS)

## Usage

### Basic Grouping
Run the script on a directory to see files grouped by time:
```bash
python3 ls_parser.py /path/to/photos
```

### Enable Visual Filtering
Use the `-v` (verbose) flag to see processing stats and filtering results. You can choose between two similarity methods:
- `orb` (default): FAST feature matching. Good for near-duplicates.
- `phash`: Perceptual hashing (dHash). Very fast and robust for similar images.

```bash
python3 ls_parser.py /path/to/photos --method phash -v
```

Sample Output:
```
Files found: 4
Groups: 1 (4)
Unique files: 2
-rw-r--r--@ 1 user  staff   9519 Feb 19 13:00:00 2026 img1.png
-rw-r--r--@ 1 user  staff  44724 Feb 19 13:00:15 2026 img4.png
```

### HEIC to JPG Conversion
If you have HEIC photos, convert them first to enable visual similarity checking:
```bash
python3 ls_parser.py /path/to/photos --convert-heic -v
```
This creates a `jpg/` subdirectory and populates it with converted images. Both `orb` and `phash` methods will automatically use these converted JPGs when processing HEIC files.

### Copy Unique Files
Copy the filtered "best" images to a `unique/` subdirectory:
```bash
python3 ls_parser.py /path/to/photos --copy
```

### JSON Output
Get structured data for use in other scripts:
```bash
python3 ls_parser.py /path/to/photos --json
```

## Configuration

- `--method`: Choose between `orb` and `phash`.
- `--threshold`: Adjust the visual similarity sensitivity. 
  - For `orb`: The minimum number of good matches (default: 10).
  - For `phash`: The maximum Hamming distance (default: 10, lower is more similar).
- `--convert-heic`: Safety first! The script will exit if a `jpg/` directory already exists to prevent accidental overwrites.

