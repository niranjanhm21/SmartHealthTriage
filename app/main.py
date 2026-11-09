import os
import subprocess
import sys

if __name__ == "__main__":
    print("Launching SmartHealth Triage UI...")
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(app_dir, ".."))
    app_py_path = os.path.join(app_dir, "app.py")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_py_path], cwd=project_root)