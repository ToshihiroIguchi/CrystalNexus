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
import signal
from pathlib import Path

# Environment-aware configuration (same as main.py)
HOST = os.getenv('CRYSTALNEXUS_HOST', '0.0.0.0')
PORT = int(os.getenv('CRYSTALNEXUS_PORT', '8080'))
DEBUG = os.getenv('CRYSTALNEXUS_DEBUG', 'False').lower() == 'true'

HEALTH_URL = f"http://localhost:{PORT}/health"
MAX_STARTUP_WAIT = 30  # seconds

# Global shutdown flag
shutdown_requested = False
server_process = None

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
    KeyboardInterrupt対応版: interruptible sleepを使用
    """
    global shutdown_requested
    
    while not shutdown_requested:
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
            
            # Interruptible sleep: 30秒を1秒刻みに分割
            for _ in range(30):
                if shutdown_requested:
                    break
                time.sleep(1)
            
        except Exception as e:
            status_queue.put(("monitor_error", f"Monitor thread error: {e}"))
            # エラー時も interruptible sleep
            for _ in range(60):
                if shutdown_requested:
                    break
                time.sleep(1)

def print_status_updates(status_queue):
    """
    対策3: ステータス更新の非同期表示
    KeyboardInterrupt対応版: interruptible sleepを使用
    """
    global shutdown_requested
    last_health_ok = time.time()
    
    while not shutdown_requested:
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
            
            # Interruptible sleep: 5秒を1秒刻みに分割
            for _ in range(5):
                if shutdown_requested:
                    break
                time.sleep(1)
            
        except Exception as e:
            print(f"Status monitor error: {e}")
            # エラー時も interruptible sleep
            for _ in range(30):
                if shutdown_requested:
                    break
                time.sleep(1)

def signal_handler(signum, frame):
    """シグナルハンドラ: Ctrl+C対応"""
    global shutdown_requested, server_process
    print(f"\nReceived signal {signum}")
    print("Shutdown requested...")
    shutdown_requested = True
    
    # サーバープロセスを即座に終了
    if server_process:
        print("Terminating server process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            print("Server terminated successfully")
        except subprocess.TimeoutExpired:
            print("Force killing server...")
            server_process.kill()
            server_process.wait()

def main():
    """Main startup routine"""
    global server_process, shutdown_requested
    
    print("CrystalNexus Startup Script")
    print("=" * 40)
    
    # シグナルハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)
    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if backend is already running
    if check_backend_status():
        print("Backend is already running!")
        print("Stopping existing server and restarting...")
        stop_existing_server()
        # Wait a moment for the port to be freed
        time.sleep(2)
    
    # Try to start the backend
    process = start_backend()
    server_process = process  # グローバル変数に保存
    
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
        # Keep the script running and wait for shutdown signal
        while not shutdown_requested:
            try:
                # 短い間隔でプロセス状態をチェック
                if process.poll() is not None:
                    print("Server process terminated unexpectedly")
                    break
                time.sleep(1)
            except KeyboardInterrupt:
                # 念のため追加のCtrl+C処理
                shutdown_requested = True
                break
                
        print("OK Main loop exited")
        
    except Exception as e:
        print(f"ERROR in main loop: {e}")
        shutdown_requested = True
        
    # 最終クリーンアップ
    if process and process.poll() is None:
        print("Final cleanup: terminating server...")
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

if __name__ == "__main__":
    main()