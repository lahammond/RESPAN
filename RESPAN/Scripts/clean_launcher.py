# clean_launcher.py
import sys
import os
import site
import subprocess
from pathlib import Path


def clean_environment():
    """Clean Python environment while preserving critical application variables"""
    # Remove user site-packages
    paths_to_remove = []
    for path in sys.path:
        if ('AppData' in path and 'site-packages' in path) or 'Roaming\\Python' in path:
            paths_to_remove.append(path)

    for path in paths_to_remove:
        if path in sys.path:
            sys.path.remove(path)

    # Disable user site
    site.USER_SITE = None
    site.USER_BASE = None

    print(f"     Environment checked: removed {len(paths_to_remove)} conflicting paths")

    # Log nnUNet variables
    nnunet_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    for var in nnunet_vars:
        if var in os.environ:
            print(f"     nnUNet variable: {var} = {os.environ[var]}")


if __name__ == "__main__":
    clean_environment()

    # Get the target script and remaining arguments
    target_script = sys.argv[1]
    remaining_args = sys.argv[2:]  # All arguments after the target script

    # Check if it's a batch file or Python script
    if target_script.endswith('.bat'):
        # For batch files, run as subprocess
        cmd = [target_script] + remaining_args
        #print(f"Executing batch file: {' '.join(cmd)}")

        # Run with cleaned environment
        env = os.environ.copy()
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    else:
        # For Python scripts, execute directly
        sys.argv = [target_script] + remaining_args
        #print(f"Executing Python script: {target_script}")
        exec(open(target_script).read(), {'__name__': '__main__'})