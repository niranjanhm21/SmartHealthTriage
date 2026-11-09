"""
Root Entry Point for SmartHealth Triage Application.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    print("Launching SmartHealth Triage UI...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_py_path = os.path.join(project_root, "app", "app.py")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_py_path], cwd=project_root)
