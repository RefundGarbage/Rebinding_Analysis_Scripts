import os
import subprocess

# Directory containing the scripts
scripts_directory = "D:\\Microscopy\\githubSc\\Analysis_steven\\Rebinding_Analysis_Scripts\\COL_BD_Lifetime_Only"

# List of scripts to run in order
scripts = ["track-sorting.py", "cell-info.py", "bound-classification.py", "gaps-and-fixes.py", "rebind-analysis.py"]

for script in scripts:
    # Construct the full path to the script
    script_path = os.path.join(scripts_directory, script)

    # Run each script using subprocess
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")