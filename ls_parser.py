import subprocess
import json
import re
import argparse
import sys

def parse_ls_l(directory="."):
    """
    Runs 'ls -l' on the specified directory and parses the output 
    into a list of dictionaries.
    """
    try:
        # Run ls -l command on the specified directory
        result = subprocess.run(['ls', '-l', directory], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        parsed_files = []
        
        # Split output into lines
        lines = output.split('\n')
        
        # Regex to parse ls -l output
        # Example line: -rw-r--r--   1 user  group   123 Feb 19 12:00 filename.txt
        # Note: This regex assumes a standard BSD/GNU ls format. 
        # It handles the variable number of spaces between columns.
        # Columns: perms, links, owner, group, size, month, day, time/year, filename
        # Perms can end with @ (extended attributes) or + (ACLs) on macOS/Linux
        # File types: - (regular), d (dir), l (link), b (block), c (char), p (pipe), s (socket)
        ls_pattern = re.compile(r'^([drwxstlbcps-]{10}[@+]?)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+\s+\S+\s+\S+)\s+(.+)$')

        for line in lines:
            # Skip total line (usually the first line)
            if line.startswith('total'):
                continue
                
            match = ls_pattern.match(line)
            if match:
                file_info = {
                    'permissions': match.group(1),
                    'links': int(match.group(2)),
                    'owner': match.group(3),
                    'group': match.group(4),
                    'size': int(match.group(5)),
                    'date': match.group(6),
                    'name': match.group(7)
                }
                parsed_files.append(file_info)
            else:
                # Debugging: print lines that didn't match (optional)
                # print(f"Skipping line: {line}", file=sys.stderr)
                pass
                
        return parsed_files

    except subprocess.CalledProcessError as e:
        print(f"Error running ls: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse 'ls -l' output into JSON.")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to parse (default: current directory)")
    args = parser.parse_args()

    files = parse_ls_l(args.directory)
    print(json.dumps(files, indent=2))
