import sys
import os
import subprocess

def main():
    """
    NKUST AoV Tool Launcher
    
    This script serves as a simple entry point to launch the Streamlit application.
    It executes: python -m streamlit run aov_app.py
    """
    
    # Get the absolute path of the current script directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "aov_app.py")

    print("===================================================")
    print("       NKUST AoV Tool - Python Launcher")
    print("===================================================")
    print(f"🚀 Launching application...")
    print(f"📂 Working Directory: {base_dir}")
    print(f"🐍 Python Executable: {sys.executable}")
    print("---------------------------------------------------")
    print("Press Ctrl+C to stop the server.")
    print("---------------------------------------------------")

    # Command to run streamlit
    # Using sys.executable ensures we use the same python interpreter (venv)
    cmd = [sys.executable, "-m", "streamlit", "run", "aov_app.py"]

    try:
        # Run the command
        # check=True will raise CalledProcessError if the command fails
        subprocess.run(cmd, cwd=base_dir, check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Application crashed with exit code: {e.returncode}")
        print("Tip: Check the error messages above for details.")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
