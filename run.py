import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")

def check_imports():
    """Check if all imports work"""
    try:
        import streamlit
        import pandas
        import sklearn
        import plotly
        import seaborn
        import numpy
        import matplotlib
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main application runner"""
    print("🚀 Cyber Threat Detection Framework")
    print("=" * 50)
    
    # Check imports
    if not check_imports():
        print("📦 Installing requirements...")
        install_requirements()
    
    # Run the application
    print("🌐 Starting Streamlit application...")
    print("📍 Open: http://localhost:8501")
    print("🔐 Demo Logins:")
    print("   - Admin: admin / admin123")
    print("   - Analyst: analyst / analyst123") 
    print("   - Viewer: viewer / viewer123")
    print("🛡️ New Features:")
    print("   - Breach Detection & Protection")
    print("   - Real-time Security Alerts")
    print("   - Automated Threat Response")
    print("   - Enhanced Authentication Security")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")

if __name__ == "__main__":
    main()