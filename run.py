"""
Launcher script to run the Gradio web application.
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the path to the app.py file
    root_dir = Path(__file__).parent
    app_path = root_dir / "webapp" / "gradio-demo" / "src" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)
    
    # Run the app using the current Python interpreter
    subprocess.run([sys.executable, str(app_path)])
