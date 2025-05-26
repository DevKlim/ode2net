import os

# Any folders here will be shown but not recursed into
SKIP_DIRS = {
    'lakh_clean',       # Large dataset folder
    'aria_midi',        # Potentially large dataset folder
    'venv',             # Virtual environment
    '.venv',            # Common virtual environment name
    '.git',             # Git repository data
    '__pycache__',      # Python cache files
    '.vscode',          # VS Code settings
    'outputs',          # Typically for generated files
    'processed_text',   # Processed data, can be large
    'processed_midi',   # Processed data, can be large
    'raw_midi',             # To avoid printing its sub-folders if they are too many and already listed in SKIP_DIRS
    '.DS_Store'         # macOS specific
}

# Files to skip printing
SKIP_FILES = {
    '.DS_Store'
}


def print_tree(root, indent='', level=0, max_level=5):
    if level > max_level:
        print(f"{indent}...") # Indicate deeper structure exists but not shown
        return

    try:
        # Sort items, directories first, then files, alphabetically
        items = sorted(os.listdir(root))
    except FileNotFoundError:
        print(f"{indent}Error: Directory not found - {root}")
        return
    except PermissionError:
        print(f"{indent}Error: Permission denied - {root}")
        return
        
    # Separate dirs and files to print dirs first
    dirs = [item for item in items if os.path.isdir(os.path.join(root, item))]
    files = [item for item in items if os.path.isfile(os.path.join(root, item))]

    # Print directories
    for name in dirs:
        if name in SKIP_DIRS and name != 'data': # Special handling for 'data' if we want to see its top level
            print(f"{indent}├── {name}/ (skipped)")
        else:
            print(f"{indent}├── {name}/")
            if name not in SKIP_DIRS : # Recurse if not in SKIP_DIRS
                 print_tree(os.path.join(root, name), indent + '│   ', level + 1, max_level)
            elif name == 'data' and 'data' not in SKIP_DIRS: # If 'data' itself is not skipped, recurse
                 print_tree(os.path.join(root, name), indent + '│   ', level + 1, max_level)


    # Print files
    for i, name in enumerate(files):
        if name in SKIP_FILES:
            continue
        connector = '└── ' if i == len(files) - 1 and not dirs else '├── ' # Adjust connector if it's the last item overall
        print(f"{indent}{connector}{name}")


if __name__ == '__main__':
    # print from current directory
    current_dir = os.path.abspath('.')
    print(f"{os.path.basename(current_dir)}/")
    # Adjust max_level if you want to see deeper or shallower structures.
    print_tree('.', indent='    ', max_level=3)