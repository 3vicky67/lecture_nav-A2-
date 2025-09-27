#!/usr/bin/env python3
"""
Ultra-fast startup script for Video RAG
"""

import subprocess
import sys
import os
import time

def main():
    # Change to the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    print("🚀 Starting ULTRA-FAST Video RAG Server...")
    print("⚡ Optimizations enabled:")
    print("   - Tiny Whisper model (fastest)")
    print("   - Lightweight embedding model")
    print("   - GPU acceleration (if available)")
    print("   - Model caching")
    print("   - Batch processing")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run the fast Flask server
        subprocess.run([sys.executable, "server_fast.py"], check=True)
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n👋 Server stopped after {elapsed:.2f} seconds")
    except Exception as e:
        print(f"❌ Error running server: {e}")

if __name__ == "__main__":
    main()
