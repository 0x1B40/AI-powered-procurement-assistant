#!/usr/bin/env python3
"""
Local development runner for the Procurement Assistant.

This script provides an easy way to run the application locally
using the virtual environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the procurement assistant locally."""

    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment!")
        print("   It's recommended to use a virtual environment for local development.")
        print("   Run: python -m venv .venv && .venv\\Scripts\\activate (Windows)")
        print("   Or:  source .venv/bin/activate (Unix/Mac)")
        print()

    # Check if .env file exists
    if not Path('.env').exists():
        print("📝 Creating .env file from template...")
        for candidate in ("example.env", "env.example"):
            env_template = Path(candidate)
            if env_template.exists():
                Path('.env').write_text(env_template.read_text())
                print(f"   Created .env from {candidate}. Please edit it with your configuration.")
                break
        else:
            print("   No env template found. Please create .env manually.")

    print("🚀 Starting Procurement Assistant locally...")
    print("   FastAPI server will be available at: http://localhost:8000")
    print("   API documentation at: http://localhost:8000/docs")
    print("   Press Ctrl+C to stop")
    print()

    try:
        # Run the FastAPI server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.api.server:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running server: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
