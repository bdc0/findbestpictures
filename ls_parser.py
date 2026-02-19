import subprocess
import json
import re

def parse_ls_l():
    """
    Runs 'ls -l' and parses the output into a list of dictionaries.
    """
    try:
        # Run ls -l command
        result = subprocess.run(['ls', '-l'], capture_output=True, text=True, check=True)
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
        ls_pattern = re.compile(r'^([drwxst-]{10}[@+]?)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+\s+\S+\s+\S+)\s+(.+)$')

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
        print(f"Error running ls: {e}")
        return []

if __name__ == "__main__":
    files = parse_ls_l()
    print(json.dumps(files, indent=2))
