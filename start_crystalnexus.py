#!/usr/bin/env python3
"""
CrystalNexus startup script
Checks if backend is running and starts it if necessary
Compatible with main.py environment configuration
"""

import requests
import subprocess
import time
import sys
import os
import platform
from pathlib import Path

# Environment-aware configuration (same as main.py)
HOST = os.getenv('CRYSTALNEXUS_HOST', '0.0.0.0')
PORT = int(os.getenv('CRYSTALNEXUS_PORT', '8080'))
DEBUG = os.getenv('CRYSTALNEXUS_DEBUG', 'False').lower() == 'true'

HEALTH_URL = f"http://localhost:{PORT}/health"
MAX_STARTUP_WAIT = 30  # seconds

def check_backend_status():
    """Check if the backend is already running"""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("service") == "CrystalNexus":
                return True
    except requests.exceptions.RequestException:
        pass
    return False

def stop_existing_server():
    """Stop existing CrystalNexus server if running"""
    try:
        import subprocess
        # Get process using port 8080 using netstat
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, text=True, shell=True
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if f':{PORT}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) > 4:
                    pid = parts[-1]
                    if pid != '0':  # Skip system processes
                        print(f"Found existing server process (PID: {pid})")
                        print("Stopping existing server...")
                        # Use PowerShell to kill process (more reliable than taskkill)
                        kill_result = subprocess.run([
                            'powershell', '-Command', f'Stop-Process -Id {pid} -Force'
                        ], capture_output=True, text=True)
                        
                        if kill_result.returncode == 0:
                            print("OK Existing server stopped")
                            return True
                        else:
                            print(f"ERROR Failed to stop process: {kill_result.stderr}")
        
        print("No existing server found to stop")
        return False
        
    except Exception as e:
        print(f"ERROR Failed to stop existing server: {e}")
        return False

def start_backend():
    """Start the FastAPI backend with environment-aware configuration"""
    print("Starting CrystalNexus backend...")
    print(f"Configuration: HOST={HOST}, PORT={PORT}, DEBUG={DEBUG}")
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Build command with environment-aware options
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", HOST,
            "--port", str(PORT)
        ]
        
        # Add reload flag only in debug mode
        if DEBUG:
            cmd.append("--reload")
            print("Debug mode: Auto-reload enabled")
        
        # Start the server as a subprocess
        # Windows対応: CREATE_NO_WINDOWフラグを設定
        kwargs = {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE}
        if platform.system() == "Windows":
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        
        process = subprocess.Popen(cmd, **kwargs)
        
        # Wait for startup
        print(f"Waiting for backend to start on port {PORT}...")
        
        for i in range(MAX_STARTUP_WAIT):
            time.sleep(1)
            if check_backend_status():
                print(f"OK Backend started successfully!")
                print(f"OK CrystalNexus is now available at http://localhost:{PORT}")
                return process
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print("ERROR Backend failed to start!")
                print("STDOUT:", stdout.decode())
                print("STDERR:", stderr.decode())
                return None
                
        print("ERROR Backend startup timeout!")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"ERROR Error starting backend: {e}")
        return None

def main():
    """Main startup routine"""
    print("CrystalNexus Startup Script")
    print("=" * 40)
    
    # Check if backend is already running
    if check_backend_status():
        print("Backend is already running!")
        print("Stopping existing server and restarting...")
        stop_existing_server()
        # Wait a moment for the port to be freed
        import time
        time.sleep(2)
    
    # Try to start the backend
    process = start_backend()
    
    if process is None:
        print("\nERROR Failed to start CrystalNexus backend")
        print("Please check the following:")
        print("1. All dependencies are installed (pip install -r requirements.txt)")
        print("2. Port 8080 is not in use by another application")
        print("3. Python environment has the necessary permissions")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("CrystalNexus is ready!")
    print(f"Open your browser and go to: http://localhost:{PORT}")
    if HOST == "0.0.0.0":
        print(f"Network access: http://<your-ip>:{PORT}")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        # Keep the script running
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down CrystalNexus...")
        process.terminate()
        process.wait()
        print("OK Server stopped")

if __name__ == "__main__":
    main()