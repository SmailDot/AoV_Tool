import os
import sys
import subprocess

def main():
    """
    Wrapper script to launch the NKUST AoV Tool.
    Users can just run 'python run_tool.py' instead of the long streamlit command.
    """
    # Get the absolute path of the current directory and the app file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "aov_app.py")
    
    print("Starting NKUST AoV Tool...")
    print(f"App Path: {app_path}")
    
    # Construct the command: streamlit run aov_app.py
    # We use sys.executable to ensure we use the same python interpreter
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError launching app: {e}")
        print("Try running manually: python -m streamlit run aov_app.py")

if __name__ == "__main__":
    main()
