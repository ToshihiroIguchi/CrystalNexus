# CrystalNexus

CrystalNexus is a comprehensive web-based application for crystal structure analysis and materials science research. It bridges the gap between traditional crystallography and modern AI by integrating the **CHGNet** (Crystal Hamiltonian Graph Neural Network) machine learning model. This allows researchers to perform real-time structure relaxation, energy prediction, and magnetic property analysis directly in the browser, without the need for complex command-line tools or expensive commercial software.

## Key Features

###  Crystal Structure Analysis
*   **Universal CIF Support**: Seamlessly load, parse, and visualizes standard Crystallographic Information Files (CIF) from any source.
*   **Interactive 3D Visualization**: powered by **3Dmol.js**, allowing you to rotate, zoom, and inspect atomic structures, bonds, and unit cells in real-time.
*   **Smart Supercell Generation**: Create custom supercells (e.g., 2x2x2) with a single click. The application automatically handles atomic positions and lattice vectors.
*   **Materials Project Integration**: Quickly search for and download structures from the Materials Project database via a direct integrated link.

###  Machine Learning Integration (CHGNet v0.4.0)
*   **State-of-the-Art Model**: Utilizes the latest CHGNet pre-trained transformer model for accurate universal interatomic potential predictions.
*   **One-Click Relaxation**: Automatically optimizes atomic positions and lattice parameters to find the ground-state structure.
*   **Property Prediction**: Instantly calculates:
    *   **Total Energy** (eV/atom)
    *   **Magnetic Moments** (magmom)
    *   **Site-Specific Energies**
*   **Auto Mode Optimization**: An advanced AI-driven feature that iteratively substitutes, deletes, or inserts atoms to discover the most energetically favorable configuration.
*   **Real-time Feedback**: Watch energy minimization progress live via dynamic sparkline charts.

###  Comprehensive Analysis Tools
*   **Local Analytics Dashboard**: A built-in SQLite database tracks your usage history and calculation statistics. **Data privacy is paramount**: all analytics are stored locally on your machine and are never uploaded to the cloud.
*   **Detailed Metrics**: Inspect precise lattice parameters (a, b, c, alpha, beta, gamma), stress tensors, and atomic forces for every step of the relaxation.
*   **Full Data Export**: Download a comprehensive ZIP archive containing:
    *   The final relaxed structure (CIF)
    *   Property data (CSV/TXT)
    *   Full optimization trajectory logs

## Directory Structure

```
CrystalNexus/
+--- main.py                 # Core FastAPI backend application
+--- start_crystalnexus.py   # robust server startup script with auto-recovery and health monitoring
+--- analytics_db.py         # Local analytics database manager (SQLite)
+--- sample_cif/            # Curated library of sample crystal structures (Gases, Metals, Oxides)
+--- templates/             # Jinja2 HTML templates (index.html, analytics.html)
+--- static/               # Static assets (js/utils.js, js/analytics.js, js/auto_mode_chart.js)
+--- tests/                # Pytest test suite (conftest.py, test_main.py, test_security.py, test_endpoints.py)
+--- uploads/              # Temporary directory for user uploads (auto-cleaned)
+--- pytest.ini             # Pytest configuration
+--- requirements.txt        # Detailed Python dependencies list
+--- CLAUDE.md               # Claude Code project instructions
```

## Installation

### Prerequisites
*   **Python**: Version 3.8 to 3.12 is required.
*   **Git**: For version control and cloning the repository.
*   **Visual C++ Build Tools (Windows Only)**: Required for compiling some Python dependencies (like `pymatgen` and `numpy`). You can download them from [Microsoft's website](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### Step-by-Step Installation Guide

#### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/ToshihiroIguchi/CrystalNexus.git
cd CrystalNexus
```

#### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to avoid conflicts with other Python projects.

**For Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*Note: If you get a permission error, run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` first.*

**For Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
Install all required libraries using `pip`. This step may take a few minutes as it installs large machine learning libraries (PyTorch, CHGNet).

```bash
pip install -r requirements.txt
```

#### 4. Verify Installation
Check if the installation was successful by listing the installed packages:
```bash
pip list
```
Ensure `chgnet`, `pymatgen`, `fastapi`, and `torch` are present.

---

## Usage Manual

### Starting the Server
We provide a dedicated startup script that handles port checking, health monitoring, and auto-restarts.

Run the following command in your terminal:
```bash
python start_crystalnexus.py
```
The script auto-detects the project's local `venv` and relaunches itself under it if present, so manually activating the virtual environment first is optional for this command.

**Manual start (without the startup script):**
If you prefer to manage the environment yourself, activate the virtual environment (see step 2 above) and run `uvicorn` directly:

```bash
uvicorn main:app --host 127.0.0.1 --port 8090
```
This skips the port-checking, health-monitoring, and auto-restart features of `start_crystalnexus.py`, but is useful for debugging or when you want full control over the process (e.g., attaching a debugger).

*   **What happens next?**
    *   The script checks if port `8090` is free.
    *   It starts the FastAPI backend server.
    *   It monitors the server health continuously.
    *   Once ready, it prints `Local: http://127.0.0.1:{PORT}`. In LAN mode (see below) it additionally prints a `Network: http://<lan-ip>:{PORT}` line and a security warning.

#### LAN access

By default the server binds to `127.0.0.1` (loopback), so it is only reachable from your own machine. To expose it to other devices on your network, run:

```bash
python start_crystalnexus.py --lan
```

This binds `0.0.0.0` and:
*   Detects and prints your machine's LAN IP address(es), so you can share the `http://<lan-ip>:{PORT}` URL with other devices.
*   Auto-generates a fresh random analytics dashboard token for this run (unless `CRYSTALNEXUS_ANALYTICS_TOKEN` is already set in your environment, in which case that value is used unchanged) and prints ready-to-use dashboard URLs with the token attached.
*   Prints a security warning summarizing the risks (see below).
*   On Windows, performs a **read-only** check of whether an inbound firewall rule exists for the port, and tells you the exact command to add one if not.

**Host precedence**: `--lan` (binds `0.0.0.0`) beats the `CRYSTALNEXUS_HOST` environment variable, which beats the `127.0.0.1` default. Setting `CRYSTALNEXUS_HOST=0.0.0.0` by hand (without `--lan`) triggers the exact same LAN-mode behavior — the warning banner, token auto-generation, and firewall check — since the script decides "LAN mode" from the resolved bind address, not from whether `--lan` was literally passed. The port can be changed with `--port PORT` (equivalent to setting `CRYSTALNEXUS_PORT`). Use `--no-firewall-check` to skip the Windows Firewall inspection.

**Security implications of LAN mode** — this is a genuinely unauthenticated app, so only expose it on a network you trust:
*   There is no authentication on `/` or on any `/api/*` endpoint; anyone who can reach the machine on that port can use the full application.
*   Any LAN device can upload CIF files and trigger CHGNet relaxations, consuming this machine's CPU and RAM. `POST /api/*` requests are rate-limited per client IP (60 requests/min, burst 20), but this only throttles abuse, it doesn't prevent it.
*   Sessions are shared globally across all clients (`MAX_SESSIONS=100`, least-recently-used eviction), so concurrent LAN users can evict each other's in-progress work.
*   Every visitor's IP address and User-Agent is logged to `analytics.db`, retained for `ANALYTICS_RETENTION_DAYS` (default 90) days.

**Windows Firewall**: the check the script performs is read-only — it never modifies firewall state itself. If no inbound allow rule is found for the port, it prints the exact command to add one, which you run yourself in an **elevated** PowerShell or Command Prompt:

```
netsh advfirewall firewall add rule name="CrystalNexus 8090" dir=in action=allow protocol=TCP localport=8090 profile=private
```

(`profile=private` keeps the rule off public/untrusted networks.) The script also prints the matching `delete rule` command so you can remove it later.

LAN IP detection is best-effort: it may list several candidate addresses if your machine has VPN/WSL/Hyper-V virtual adapters, and it may occasionally fail to detect anything, in which case run `ipconfig` and look for your Wi-Fi/Ethernet adapter's IPv4 address.

### Workflow Guide

#### 1. Loading a Structure
*   **Option A (Select Sample)**: Click the **"Select CIF File"** button. Choose a file from the dropdown list (e.g., `Oxides/BaTiO3(cubic).cif`).
*   **Option B (Upload)**: Click **"Upload CIF"** to use your own file.
*   **Option C (Materials Project)**: Click the link in the modal to search the Materials Project database, download a CIF, and then upload it here.

#### 2. Visualization
*   **Rotate**: Left-click and drag.
*   **Zoom**: Mouse wheel scroll.
*   **Pan**: Right-click (or Ctrl+Left-click) and drag.
*   **Inspect**: Hover over atoms to see their element and coordinates.

#### 3. Structure Analysis (CHGNet)
*   **Analyze & Relax**: Click **"Analyze Structure"**. CHGNet relaxes the geometry (optimizing atomic positions and the cell) and reports the energy of the resulting structure. The 3D view updates to show the relaxed structure.
*   **Advanced Settings**: Expand **"⚙️ Advanced Settings"** to tune the force tolerance, max steps, and optimizer (LBFGS/FIRE/BFGS) before analyzing.

#### 4. Advanced Editing
*   **Supercell**: Open the **"Structure Operations"** menu. Enter dimensions (e.g., 2 2 2) and click **"Create Supercell"**.
*   **Structure Modifier**: Use the Tabs UI to manipulate the crystal:
    *   **Edit Existing**: Substitute or delete specific atoms.
    *   **Insert New**: Add new atoms into stable void sites (interstitial sites) within the crystal lattice.
*   **Auto Mode**: Select "Auto Mode" (within Edit or Insert) to let AI iteratively improve your structure by testing the most energetically favorable atomic changes.

#### 5. Exporting Results
*   After an analysis is complete, click the **"Detailed Analysis"** button.
*   In the modal, click **"Download All Data (ZIP)"** to save your work.

#### 6. View Analytics
*   Access the local analytics dashboard at `http://localhost:8090/analytics` to monitor:
    *   Daily usage trends
    *   Most analyzed structures
    *   Recent calculation logs
*   By default the dashboard is loopback-only: no token is needed when accessing it from `127.0.0.1`/`localhost`. Once `CRYSTALNEXUS_ANALYTICS_TOKEN` is set — including automatically when you run `python start_crystalnexus.py --lan` — access from anywhere, including localhost, requires that token, either as a `?token=...` query parameter or an `X-Analytics-Token` header. When `--lan` auto-generates a token, it prints ready-to-use `?token=...` URLs for both `127.0.0.1` and your LAN IP.

---

## Running Tests

The test suite lives in the `tests/` directory and uses **pytest** (included in `requirements.txt`).

Run the full suite from the repository root:

**For Windows:**
```powershell
venv\Scripts\python.exe -m pytest tests/ -v
```

**For macOS / Linux (with the virtual environment activated):**
```bash
python -m pytest tests/ -v
```

For a faster feedback loop, skip the slow tests:
```bash
python -m pytest tests/ -m "not slow"
```

---

## Troubleshooting

### Q: "Backend failed to start" or "Port 8090 is in use"
*   **Solution**: The `start_crystalnexus.py` script automatically detects and stops any existing process using port 8090. Simply running `python start_crystalnexus.py` again should resolve the issue. Manual intervention is only necessary in rare cases where the script lacks permission to terminate the process.

### Q: "Module not found: chgnet"
*   **Correction**: Ensure you activated your virtual environment (`venv`) before running the server. Re-run `pip install -r requirements.txt`.

### Q: "DLL load failed" on Windows
*   **Correction**: This usually means a missing system dependency. Install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

### Q: Performance is slow
*   **Correction**: Computing relaxation for large supercells (100+ atoms) can be slow because CrystalNexus is configured to run on **CPU only** to ensure broad compatibility and stability. The relaxation algorithm was changed from FIRE to LBFGS, which improves performance, but large structures on CPU are still inherently slower than on GPU. Please be patient when analyzing complex structures or large supercells.
*   **Guardrails**: To keep long-running relaxations manageable, a timeout returns the best structure found so far if relaxation does not converge in time, and limits on atom count, step count, and force tolerance reject requests that would be too large or resource-intensive to process.

## Technology Stack

*   **Backend**: FastAPI, Uvicorn, Python 3.8+
*   **Machine Learning**: CHGNet, PyTorch, Pymatgen
*   **Frontend**: HTML5, Vanilla JS, Jinja2, 3Dmol.js
*   **Database**: SQLite (for local analytics)

## License

MIT License - Copyright (c) 2025 Toshihiro Iguchi

## Citation

If you use CrystalNexus in your research, please cite:

```bibtex
@software{crystalnexus2025,
  title={CrystalNexus: Web-based Crystal Structure Analysis with Machine Learning},
  author={Toshihiro Iguchi},
  year={2025},
  url={https://github.com/ToshihiroIguchi/CrystalNexus}
}
```
