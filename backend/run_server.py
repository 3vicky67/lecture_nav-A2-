#!/usr/bin/env python3
"""
Simple script to run the Flask backend server
"""

import subprocess
import sys
import os

def main():
    # Change to the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    print("🚀 Starting Video Search Backend Server...")
    print("📍 Server will be available at: http://localhost:5000")
    print("🔍 API Endpoints:")
    print("   - POST /api/ingest_video - Upload and transcribe video")
    print("   - POST /api/search_timestamps - Search video content")
    print("=" * 50)
    
    try:
        # Run the Flask server
        subprocess.run([sys.executable, "server.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error running server: {e}")

if __name__ == "__main__":
    main()
