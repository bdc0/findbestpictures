import subprocess
import json
import re
import argparse
import sys
from datetime import datetime, timedelta

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
                
            match = ls_pattern.match(line)
            if match:
                date_str = match.group(6)
                # Parse date: Feb 19 12:00:00 2026
                dt_object = datetime.strptime(date_str, "%b %d %H:%M:%S %Y")
                
                file_info = {
                    'permissions': match.group(1),
                    'links': int(match.group(2)),
                    'owner': match.group(3),
                    'group': match.group(4),
                    'size': int(match.group(5)),
                    'date': date_str,
                    'datetime': dt_object.isoformat(), # Store as ISO for JSON
                    'timestamp': dt_object.timestamp(), # Store timestamp for sorting/calc
                    'name': match.group(7),
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse 'ls -l' output and group by time.")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to parse (default: current directory)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format instead of original ls lines")
    args = parser.parse_args()

    files = parse_ls_l(args.directory)
    groups = group_files_by_time(files)
    
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
