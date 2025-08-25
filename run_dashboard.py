#!/usr/bin/env python3
"""
Simple launcher for the Streamlit dashboard.
Run this to start the web interface.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    try:
        # Change to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Launch Streamlit
        print("🚀 Starting Grid Trading Strategy Dashboard...")
        print("📊 The dashboard will open in your web browser")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thanks for using the Grid Trading Strategy tool!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Make sure you have streamlit installed: uv add streamlit")

if __name__ == "__main__":
    main()
