#!/usr/bin/env python3
"""
Prepare the project for deployment by checking requirements and providing guidance.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_file_exists(filename, description):
    """Check if a required file exists"""
    if os.path.exists(filename):
        print(f"✅ {description}: {filename}")
        return True
    else:
        print(f"❌ Missing {description}: {filename}")
        return False

def check_git_status():
    """Check git status and provide guidance"""
    try:
        # Check if git is initialized
        subprocess.run(["git", "status"], capture_output=True, check=True)
        print("✅ Git repository initialized")
        
        # Check for uncommitted changes
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  Uncommitted changes detected:")
            print(result.stdout)
            print("💡 Run: git add . && git commit -m 'Prepare for deployment'")
        else:
            print("✅ No uncommitted changes")
            
    except subprocess.CalledProcessError:
        print("❌ Git not initialized")
        print("💡 Run: git init && git add . && git commit -m 'Initial commit'")

def check_streamlit_app():
    """Check if Streamlit app can be imported"""
    try:
        # Try to parse the streamlit app file
        with open("streamlit_app.py", "r") as f:
            content = f.read()
            if "streamlit" in content and "st.title" in content:
                print("✅ Streamlit app appears valid")
                return True
            else:
                print("⚠️  Streamlit app may have issues")
                return False
    except Exception as e:
        print(f"❌ Error reading streamlit_app.py: {e}")
        return False

def check_data_file():
    """Check if data file exists and is accessible"""
    data_path = "data/eth-prices.csv"
    if os.path.exists(data_path):
        try:
            # Check file size
            size = os.path.getsize(data_path)
            print(f"✅ Data file exists: {data_path} ({size:,} bytes)")
            
            # Check if it's readable
            with open(data_path, 'r') as f:
                first_line = f.readline()
                if "Date" in first_line and "Value" in first_line:
                    print("✅ Data file format appears correct")
                    return True
                else:
                    print("⚠️  Data file format may be incorrect")
                    return False
        except Exception as e:
            print(f"❌ Error reading data file: {e}")
            return False
    else:
        print(f"❌ Data file missing: {data_path}")
        return False

def provide_deployment_options():
    """Show deployment options"""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT OPTIONS")
    print("="*60)
    
    print("\n1️⃣  STREAMLIT COMMUNITY CLOUD (Recommended - FREE)")
    print("   • Go to: https://share.streamlit.io")
    print("   • Connect your GitHub repository")
    print("   • Main file: streamlit_app.py")
    print("   • Deploy automatically")
    
    print("\n2️⃣  RAILWAY ($5/month)")
    print("   • Go to: https://railway.app")
    print("   • Connect GitHub repository")
    print("   • Automatic deployment")
    
    print("\n3️⃣  HEROKU (Free tier available)")
    print("   • Install Heroku CLI")
    print("   • heroku create your-app-name")
    print("   • git push heroku main")
    
    print("\n4️⃣  DOCKER + VPS")
    print("   • docker build -t crypto-trading .")
    print("   • Deploy to any VPS service")

def main():
    """Main deployment preparation check"""
    print("🔍 DEPLOYMENT READINESS CHECK")
    print("="*50)
    
    all_good = True
    
    # Check required files
    required_files = [
        ("streamlit_app.py", "Streamlit app"),
        ("requirements.txt", "Python requirements"),
        ("method-analysis.py", "Main analysis module"),
        ("data/eth-prices.csv", "ETH price data"),
        ("README.md", "Documentation"),
    ]
    
    for filename, description in required_files:
        if not check_file_exists(filename, description):
            all_good = False
    
    print("\n" + "-"*50)
    
    # Check git status
    check_git_status()
    
    print("\n" + "-"*50)
    
    # Check Streamlit app
    if not check_streamlit_app():
        all_good = False
    
    # Check data file
    if not check_data_file():
        all_good = False
    
    print("\n" + "-"*50)
    
    if all_good:
        print("🎉 READY FOR DEPLOYMENT!")
        print("All required files are present and appear valid.")
        provide_deployment_options()
        
        print("\n💡 QUICK START:")
        print("1. Push to GitHub: git push origin main")
        print("2. Go to https://share.streamlit.io")
        print("3. Connect your repo and deploy!")
        
    else:
        print("⚠️  SOME ISSUES FOUND")
        print("Please fix the issues above before deploying.")
    
    print("\n📖 For detailed deployment instructions, see: deploy_guide.md")

if __name__ == "__main__":
    main()
