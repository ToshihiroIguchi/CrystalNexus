#!/usr/bin/env python3
"""
CrystalNexus startup script
Checks if backend is running and starts it if necessary
Compatible with main.py environment configuration
"""

import subprocess
import time
import sys
import os
import platform
import threading
import queue
import signal
import argparse
import secrets
import socket
import ipaddress
import collections
from pathlib import Path
from typing import Optional


def _relaunch_in_venv():
    """Re-exec this script under the project's local venv Python if it isn't already running there.

    Must run before `import requests` below, since a non-venv interpreter may not have
    third-party dependencies installed.
    """
    script_dir = Path(__file__).resolve().parent
    if platform.system() == "Windows":
        venv_python = script_dir / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = script_dir / "venv" / "bin" / "python"

    if not venv_python.exists():
        return

    try:
        already_in_venv = os.path.samefile(str(venv_python), sys.executable)
    except OSError:
        already_in_venv = False

    if already_in_venv:
        return

    print(f"Activating local virtual environment: {venv_python}")
    try:
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    except Exception as e:
        print(f"Warning: failed to relaunch inside venv ({e}); continuing with current interpreter.")


if __name__ == "__main__":
    _relaunch_in_venv()

import requests

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})


def resolve_host(lan: bool, env_host: Optional[str]) -> str:
    """CLI --lan beats the CRYSTALNEXUS_HOST env var, which beats the loopback default."""
    if lan:
        return "0.0.0.0"
    return env_host if env_host else "127.0.0.1"


def is_lan_binding(host: str) -> bool:
    """True when the bind address is reachable from other machines on the network."""
    return host not in LOOPBACK_HOSTS  # 0.0.0.0, ::, or an explicit interface IP


def health_host(host: str) -> str:
    """Which address the launcher itself should probe for /health.

    The launcher always runs on the same machine as the server, so probing
    loopback is correct and robust even when the server also binds 0.0.0.0
    (it listens on loopback too in that case). On Windows, `localhost` can
    resolve to ::1 first, and uvicorn bound to 0.0.0.0 is IPv4-only, so use
    the literal 127.0.0.1 instead of the hostname `localhost`.
    """
    if host in ("0.0.0.0", "127.0.0.1", "localhost", ""):
        return "127.0.0.1"
    if host in ("::", "::1"):
        return "[::1]"
    return host  # an explicit interface IP: uvicorn is NOT listening on loopback here


def resolve_analytics_token(lan_mode: bool) -> Optional[str]:
    """Decide what CRYSTALNEXUS_ANALYTICS_TOKEN the child server process should run with.

    Precedence:
      env var set to a non-empty value -> use it verbatim (never overwrite the user's choice)
      env var present but set to ""    -> explicit opt-out; warn when lan_mode, return None
      env var unset and lan_mode       -> generate a fresh secrets.token_urlsafe(32)
      env var unset and not lan_mode   -> None (main.py's existing loopback-only default applies)

    Not generating a token outside LAN mode is deliberate: forcing a token on
    loopback-only use would make the analytics dashboard *less* usable there
    (main.py's loopback bypass would no longer apply) for no security benefit.
    """
    raw = os.environ.get("CRYSTALNEXUS_ANALYTICS_TOKEN")
    if raw:
        return raw
    if raw is not None:  # present but empty string: deliberate opt-out
        if lan_mode:
            print("WARNING: CRYSTALNEXUS_ANALYTICS_TOKEN is set to an empty value.")
            print("         /analytics and /api/analytics/* will be reachable from")
            print("         any LAN client with no token. This exposes every visitor")
            print("         IP/User-Agent and uploaded filename/formula on your LAN.")
        return None
    if lan_mode:
        return secrets.token_urlsafe(32)
    return None


def detect_lan_ipv4():
    """Best-effort LAN IPv4 detection. Never raises. Returns (primary_or_None, other_candidates).

    Primary: connect() a UDP socket to a public IP (no packet is actually sent
    for UDP connect -- it only asks the OS routing table which local address
    would be used to reach the internet). This is preferred over
    socket.getaddrinfo(socket.gethostname()) because on a typical Windows
    machine that call often returns Hyper-V/WSL/Docker/VPN virtual adapter
    addresses ahead of the real Wi-Fi/Ethernet address, which would print a
    URL no other device on the LAN can actually reach.
    """
    primary = None
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        primary = s.getsockname()[0]
    except OSError:
        primary = None
    finally:
        s.close()

    others = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            try:
                addr = ipaddress.IPv4Address(ip)
            except ValueError:
                continue
            if addr.is_loopback or addr.is_link_local:
                continue
            if ip != primary and ip not in others:
                others.append(ip)
    except OSError:
        pass
    return primary, others


_FIREWALL_CHECK_PS = (
    "$p = {port}; "
    "Get-NetFirewallPortFilter -PolicyStore ActiveStore | "
    "Where-Object {{ $_.Protocol -eq 'TCP' -and "
    "($_.LocalPort -contains [string]$p -or $_.LocalPort -contains 'Any') }} | "
    "ForEach-Object {{ $r = $_ | Get-NetFirewallRule; "
    "if ($r.Enabled -eq 'True' -and $r.Direction -eq 'Inbound' -and "
    "$r.Action -eq 'Allow') {{ 'MATCH:' + $r.DisplayName }} }}"
)


def check_firewall_rule(port: int) -> Optional[bool]:
    """Read-only Windows Firewall inspection. True/False, or None if undetermined.

    Uses PowerShell's Get-NetFirewallRule/Get-NetFirewallPortFilter cmdlets
    (readable by a standard user on a default Windows 11 install) rather than
    parsing `netsh` text output, because netsh's localized text (e.g. on a
    Japanese Windows install) will not match an English-language pattern.
    This function must NEVER call any New-/Set-/Remove- cmdlet or `netsh ...
    add/delete` -- it only ever reads firewall state.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             _FIREWALL_CHECK_PS.format(port=port)],
            capture_output=True, text=True, errors="replace", timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return any(line.startswith("MATCH:") for line in result.stdout.splitlines())


def _child_env(host: str, port: int, token: Optional[str]) -> dict:
    """Build the environment for the uvicorn child process.

    Returns a full COPY of os.environ (never mutates the parent's os.environ)
    with HOST/PORT/token overridden. A full copy (not a partial dict) is
    required on Windows -- python -m uvicorn needs PATH/SystemRoot and other
    inherited variables to run at all. Passing this via subprocess.Popen's
    env= keyword (rather than mutating os.environ in the launcher process)
    also keeps the analytics token out of the launcher's own environment,
    which matters because the launcher itself shells out to netstat/tasklist/
    powershell elsewhere (stop_existing_server, check_firewall_rule).
    """
    env = dict(os.environ)
    env["CRYSTALNEXUS_HOST"] = host
    env["CRYSTALNEXUS_PORT"] = str(port)
    if token:
        env["CRYSTALNEXUS_ANALYTICS_TOKEN"] = token
    return env


# Environment-aware configuration (same as main.py)
# Default to loopback; set CRYSTALNEXUS_HOST=0.0.0.0 to expose on the network
HOST = os.getenv('CRYSTALNEXUS_HOST', '127.0.0.1')
PORT = int(os.getenv('CRYSTALNEXUS_PORT', '8080'))
DEBUG = os.getenv('CRYSTALNEXUS_DEBUG', 'False').lower() == 'true'

HEALTH_URL = f"http://localhost:{PORT}/health"
LAN_MODE = is_lan_binding(HOST)
ANALYTICS_TOKEN = None
MAX_STARTUP_WAIT = 30  # seconds

# Global shutdown flag
shutdown_requested = False
server_process = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="start_crystalnexus.py",
        description="Start the CrystalNexus backend (loopback by default).",
        epilog=(
            "Host precedence: --lan (binds 0.0.0.0) beats CRYSTALNEXUS_HOST, "
            "which beats the 127.0.0.1 default."
        ),
    )
    parser.add_argument(
        "--lan", action="store_true",
        help="Bind 0.0.0.0 and print the LAN URL. Exposes this unauthenticated "
             "app to every device on your network.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override CRYSTALNEXUS_PORT (default 8080).",
    )
    parser.add_argument(
        "--no-firewall-check", action="store_true",
        help="Skip the read-only Windows Firewall inspection.",
    )
    return parser.parse_args(argv)


def _apply_runtime_config(args):
    """Resolve CLI args + env into the module's runtime globals. Must run
    before check_backend_status()/stop_existing_server() are called so they
    see the final HOST/PORT."""
    global HOST, PORT, HEALTH_URL, LAN_MODE, ANALYTICS_TOKEN
    if args.port is not None:
        PORT = args.port
    HOST = resolve_host(args.lan, os.getenv("CRYSTALNEXUS_HOST"))
    LAN_MODE = is_lan_binding(HOST)
    HEALTH_URL = f"http://{health_host(HOST)}:{PORT}/health"
    ANALYTICS_TOKEN = resolve_analytics_token(LAN_MODE)


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
        # Get process using port 8080 using netstat
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, text=True, shell=True
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if 'LISTENING' not in line:
                continue
            parts = line.split()
            # netstat -ano columns: Proto, Local Address, Foreign Address, State, PID
            if len(parts) < 5:
                continue
            local_address = parts[1]
            # Match only the LOCAL address port (a foreign-address port must not match)
            if not local_address.endswith(f':{PORT}'):
                continue
            pid = parts[-1]
            if pid != '0':  # Skip system processes
                # Verify the process is actually a python/uvicorn server before killing
                task_result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}'],
                    capture_output=True, text=True
                )
                task_output = task_result.stdout.lower()
                if 'python' not in task_output and 'uvicorn' not in task_output:
                    print(f"Skipping PID {pid} on port {PORT}: not a python/uvicorn process")
                    continue
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

def _stream_child_output(process: subprocess.Popen, tail: "collections.deque[str]") -> None:
    """Relay the child's stdout/stderr to our console and keep the last lines in `tail`.

    Must start right after Popen: the OS pipe buffer is small (a few tens of
    KB on Windows), and an uvicorn process writes to it continuously, so a
    pipe nobody drains will eventually block the child. Combining
    stderr into stdout (see start_backend) means a single reader here
    captures both.
    """
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            tail.append(line)
    except Exception:
        pass


def _print_captured_output(tail: "collections.deque[str]") -> None:
    """Show the child's last output lines to help diagnose a startup failure."""
    if not tail:
        print("(No output was captured from the server process.)")
        return
    print()
    print("Captured server output (most recent lines):")
    print("-" * 40)
    for line in tail:
        print(line, end="")
    print("-" * 40)


def start_backend():
    """Start the FastAPI backend with environment-aware configuration"""
    print("Starting CrystalNexus backend...")
    token_status = "generated" if (ANALYTICS_TOKEN and not os.environ.get("CRYSTALNEXUS_ANALYTICS_TOKEN")) else (
        "from environment" if ANALYTICS_TOKEN else "none"
    )
    print(f"Configuration: HOST={HOST}, PORT={PORT}, DEBUG={DEBUG}, LAN_MODE={LAN_MODE}, ANALYTICS_TOKEN={token_status}")
    
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
        
        # Countermeasure 1: Proper handling of process output
        # Pipe stdout/stderr (merged) through a dedicated reader thread started
        # immediately below, so the pipe is continuously drained (avoiding the
        # buffer-clogging this used to dodge by not piping at all) while also
        # letting us show the child's own error output if startup fails --
        # inherited console handles do not reliably surface here on Windows.
        kwargs = {}
        if platform.system() == "Windows":
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

        print("Starting server with direct output mode...")
        process = subprocess.Popen(
            cmd,
            env=_child_env(HOST, PORT, ANALYTICS_TOKEN),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **kwargs,
        )

        output_tail = collections.deque(maxlen=200)
        output_thread = threading.Thread(
            target=_stream_child_output, args=(process, output_tail), daemon=True
        )
        output_thread.start()

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
                output_thread.join(timeout=2)
                _print_captured_output(output_tail)
                return None
                
        print("ERROR Backend startup timeout!")
        process.terminate()
        output_thread.join(timeout=2)
        _print_captured_output(output_tail)
        return None
        
    except Exception as e:
        print(f"ERROR Error starting backend: {e}")
        return None

def monitor_process_health(process, status_queue):
    """
    Countermeasure 3: Process Monitoring - Monitor process health in background
    KeyboardInterrupt compatible version: Use interruptible sleep
    """
    global shutdown_requested
    
    while not shutdown_requested:
        try:
            # Check process state
            if process.poll() is not None:
                status_queue.put(("process_died", f"Process terminated unexpectedly (exit code: {process.returncode})"))
                break
            
            # Run health check
            try:
                response = requests.get(HEALTH_URL, timeout=3)
                if response.status_code != 200:
                    status_queue.put(("health_check_failed", f"Health check failed with status {response.status_code}"))
                else:
                    status_queue.put(("health_ok", "Server responding normally"))
            except requests.exceptions.RequestException as e:
                status_queue.put(("health_check_error", f"Health check error: {e}"))
            
            # Interruptible sleep: Divide 30 seconds into 1-second increments
            for _ in range(30):
                if shutdown_requested:
                    break
                time.sleep(1)
            
        except Exception as e:
            status_queue.put(("monitor_error", f"Monitor thread error: {e}"))
            # Interruptible sleep even on error
            for _ in range(60):
                if shutdown_requested:
                    break
                time.sleep(1)

def print_status_updates(status_queue):
    """
    Countermeasure 3: Asynchronous Status Display
    KeyboardInterrupt compatible version: Use interruptible sleep
    """
    global shutdown_requested
    last_health_ok = time.time()
    
    while not shutdown_requested:
        try:
            # Non-blocking status check
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
                    # Do not output detailed logs when normal (work silently)
                elif status_type == "health_check_error":
                    if time.time() - last_health_ok > 120:  # Warn only if abnormal for more than 2 minutes
                        print(f"\n[{current_time}] WARNING: {message}")
                elif status_type == "monitor_error":
                    print(f"\n[{current_time}] ERROR: {message}")
                    
            except queue.Empty:
                pass
            
            # Interruptible sleep: Divide 5 seconds into 1-second increments
            for _ in range(5):
                if shutdown_requested:
                    break
                time.sleep(1)
            
        except Exception as e:
            print(f"Status monitor error: {e}")
            # Interruptible sleep even on error
            for _ in range(30):
                if shutdown_requested:
                    break
                time.sleep(1)

def signal_handler(signum, frame):
    """Signal Handler: Handle Ctrl+C"""
    global shutdown_requested, server_process
    print(f"\nReceived signal {signum}")
    print("Shutdown requested...")
    shutdown_requested = True
    
    # Terminate server process immediately
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

    args = parse_args()
    _apply_runtime_config(args)

    print("CrystalNexus Startup Script")
    print("=" * 40)
    
    # Set signal handlers
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
    server_process = process  # Save to global variable
    
    if process is None:
        print("\nERROR Failed to start CrystalNexus backend")
        print("Please check the following:")
        print("1. All dependencies are installed (pip install -r requirements.txt)")
        print("2. Port 8080 is not in use by another application")
        print("3. Python environment has the necessary permissions")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("CrystalNexus is ready!")
    print(f"Local: http://127.0.0.1:{PORT}")

    lan_primary_ip = None
    if LAN_MODE:
        lan_primary_ip, lan_other_ips = detect_lan_ipv4()
        if lan_primary_ip:
            print(f"Network: http://{lan_primary_ip}:{PORT}")
        else:
            print("Network: could not auto-detect a LAN IP. Run `ipconfig` and look")
            print(f"         for your Wi-Fi/Ethernet adapter's IPv4 address, then use http://<that-ip>:{PORT}")
        if lan_other_ips:
            print("Other interfaces (VPN/WSL/Hyper-V may appear here):")
            for ip in lan_other_ips:
                print(f"  http://{ip}:{PORT}")

        print()
        print("!!  LAN EXPOSURE WARNING  !!")
        print("  * There is NO authentication on / or on any /api/* endpoint.")
        print(f"    Any device that can reach this machine on port {PORT} can use the app.")
        print("  * Any LAN device can upload CIF files and start CHGNet relaxations,")
        print("    consuming this machine's CPU and RAM. Only POST /api/* is rate limited")
        print("    (60 req/min, burst 20, per client IP).")
        print("  * Sessions are shared globally, not per user: MAX_SESSIONS=100 with")
        print("    least-recently-used eviction, so one busy client can evict another")
        print("    client's in-progress structure.")
        print("  * Every visitor's IP address and User-Agent is logged to analytics.db")
        print("    (retained ANALYTICS_RETENTION_DAYS=90 days).")
        print("  * Only run this on a network you trust. Stop with Ctrl+C when finished.")

        if ANALYTICS_TOKEN and not os.environ.get("CRYSTALNEXUS_ANALYTICS_TOKEN"):
            print()
            print(f"Analytics token (generated for this run): {ANALYTICS_TOKEN}")
            print(f"  Dashboard:  http://127.0.0.1:{PORT}/analytics?token={ANALYTICS_TOKEN}")
            if lan_primary_ip:
                print(f"  From LAN:   http://{lan_primary_ip}:{PORT}/analytics?token={ANALYTICS_TOKEN}")
            print("  This token changes on every launch. Export CRYSTALNEXUS_ANALYTICS_TOKEN")
            print("  to keep it stable. The token appears in the URL, so it lands in browser history.")

        print()

    print("Press Ctrl+C to stop the server")
    print("=" * 40)

    if LAN_MODE and platform.system() == "Windows" and not args.no_firewall_check:
        fw_result = check_firewall_rule(PORT)
        if fw_result is True:
            print(f"Windows Firewall: an inbound allow rule for TCP {PORT} already exists.")
        elif fw_result is False:
            print(f"Windows Firewall: no inbound allow rule found for TCP {PORT}.")
            print("  Other devices probably cannot reach this server. To add one, run this in an")
            print("  ELEVATED PowerShell / Command Prompt (this script will not do it for you and")
            print("  does not need admin rights):")
            print()
            print(f'    netsh advfirewall firewall add rule name="CrystalNexus {PORT}" dir=in action=allow protocol=TCP localport={PORT} profile=private')
            print()
            print("  profile=private keeps the rule off public/untrusted networks.")
            print("  To remove it later:")
            print()
            print(f'    netsh advfirewall firewall delete rule name="CrystalNexus {PORT}" protocol=TCP localport={PORT}')
            print()
            print('  Note: Windows may also show its own "Allow python.exe?" dialog when uvicorn')
            print("  first binds. If it was dismissed or denied before, a block rule exists and")
            print("  must be removed in wf.msc.")
        else:
            print(f"Windows Firewall: could not determine whether port {PORT} is allowed (the check")
            print("  was skipped; this is not an error). If LAN clients cannot connect, see the")
            print("  netsh command in the README.")

    # Countermeasure 3: Start process monitoring thread
    status_queue = queue.Queue()
    
    # Background monitoring thread
    monitor_thread = threading.Thread(target=monitor_process_health, args=(process, status_queue))
    monitor_thread.daemon = True  # Terminate on main process exit
    monitor_thread.start()
    
    # Status display thread
    status_thread = threading.Thread(target=print_status_updates, args=(status_queue,))
    status_thread.daemon = True  # Terminate on main process exit
    status_thread.start()
    
    print("Background monitoring started...")
    
    try:
        # Keep the script running and wait for shutdown signal
        while not shutdown_requested:
            try:
                # Check process state at short intervals
                if process.poll() is not None:
                    print("Server process terminated unexpectedly")
                    break
                time.sleep(1)
            except KeyboardInterrupt:
                # Additional Ctrl+C handling just in case
                shutdown_requested = True
                break
                
        print("OK Main loop exited")
        
    except Exception as e:
        print(f"ERROR in main loop: {e}")
        shutdown_requested = True
        
    # Final cleanup
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