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
import threading
import queue
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
        
        # 対策1: プロセス出力の適切な処理
        # stdout/stderrをPIPEしない（コンソールに直接出力してバッファー詰まり回避）
        kwargs = {}
        if platform.system() == "Windows":
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        
        print("Starting server with direct output mode...")
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
                print("ERROR Backend failed to start!")
                print("Process terminated unexpectedly during startup")
                return None
                
        print("ERROR Backend startup timeout!")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"ERROR Error starting backend: {e}")
        return None

def monitor_process_health(process, status_queue):
    """
    対策3: プロセス監視機能 - バックグラウンドでプロセスの健全性を監視
    """
    while True:
        try:
            # プロセス状態チェック
            if process.poll() is not None:
                status_queue.put(("process_died", f"Process terminated unexpectedly (exit code: {process.returncode})"))
                break
            
            # ヘルスチェック実行
            try:
                response = requests.get(HEALTH_URL, timeout=3)
                if response.status_code != 200:
                    status_queue.put(("health_check_failed", f"Health check failed with status {response.status_code}"))
                else:
                    status_queue.put(("health_ok", "Server responding normally"))
            except requests.exceptions.RequestException as e:
                status_queue.put(("health_check_error", f"Health check error: {e}"))
            
            time.sleep(30)  # 30秒間隔で監視
            
        except Exception as e:
            status_queue.put(("monitor_error", f"Monitor thread error: {e}"))
            time.sleep(60)  # エラー時は1分待機してリトライ

def print_status_updates(status_queue):
    """
    対策3: ステータス更新の非同期表示
    """
    last_health_ok = time.time()
    
    while True:
        try:
            # ノンブロッキングでステータスチェック
            try:
                status_type, message = status_queue.get_nowait()
                current_time = time.strftime("%H:%M:%S")
                
                if status_type == "process_died":
                    print(f"\n[{current_time}] CRITICAL: {message}")
                    print("Server process has terminated!")
                elif status_type == "health_check_failed":
                    print(f"\n[{current_time}] WARNING: {message}")
                elif status_type == "health_ok":
                    last_health_ok = time.time()
                    # 正常時は詳細ログを出力しない（静かに動作）
                elif status_type == "health_check_error":
                    if time.time() - last_health_ok > 120:  # 2分以上異常が続く場合のみ警告
                        print(f"\n[{current_time}] WARNING: {message}")
                elif status_type == "monitor_error":
                    print(f"\n[{current_time}] ERROR: {message}")
                    
            except queue.Empty:
                pass
            
            time.sleep(5)  # 5秒間隔でキューチェック
            
        except Exception as e:
            print(f"Status monitor error: {e}")
            time.sleep(30)

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
    
    # 対策3: プロセス監視スレッドを開始
    status_queue = queue.Queue()
    
    # バックグラウンド監視スレッド
    monitor_thread = threading.Thread(target=monitor_process_health, args=(process, status_queue))
    monitor_thread.daemon = True  # メインプロセス終了時に自動終了
    monitor_thread.start()
    
    # ステータス表示スレッド
    status_thread = threading.Thread(target=print_status_updates, args=(status_queue,))
    status_thread.daemon = True  # メインプロセス終了時に自動終了
    status_thread.start()
    
    print("Background monitoring started...")
    
    try:
        # Keep the script running
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down CrystalNexus...")
        
        # 対策2: 強制終了機能の強化
        print("Terminating server process...")
        process.terminate()
        
        try:
            # 10秒間待機して正常終了を試行
            print("Waiting for graceful shutdown...")
            process.wait(timeout=10)
            print("OK Server stopped gracefully")
        except subprocess.TimeoutExpired:
            # 正常終了に失敗した場合は強制終了
            print("Graceful shutdown failed, forcing termination...")
            process.kill()
            try:
                process.wait(timeout=5)
                print("OK Server force-stopped")
            except subprocess.TimeoutExpired:
                print("WARNING: Server may still be running - check manually")
        except Exception as e:
            print(f"ERROR during shutdown: {e}")
            print("WARNING: Server may still be running - check manually")

if __name__ == "__main__":
    main()