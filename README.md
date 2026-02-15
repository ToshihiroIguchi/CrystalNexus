# CrystalNexus

CrystalNexus is a comprehensive web-based application for crystal structure analysis and materials science research. It bridges the gap between traditional crystallography and modern AI by integrating the **CHGNet** (Crystal Hamiltonian Graph Neural Network) machine learning model. This allows researchers to perform real-time structure relaxation, energy prediction, and magnetic property analysis directly in the browser, without the need for complex command-line tools or expensive commercial software.

## Key Features

### 🔬 Crystal Structure Analysis
*   **Universal CIF Support**: Seamlessly load, parse, and visualizes standard Crystallographic Information Files (CIF) from any source.
*   **Interactive 3D Visualization**: powered by **3Dmol.js**, allowing you to rotate, zoom, and inspect atomic structures, bonds, and unit cells in real-time.
*   **Smart Supercell Generation**: Create custom supercells (e.g., 2x2x2) with a single click. The application automatically handles atomic positions and lattice vectors.
*   **Materials Project Integration**: Quickly search for and download structures from the Materials Project database via a direct integrated link.

### 🤖 Machine Learning Integration (CHGNet v0.4.0)
*   **State-of-the-Art Model**: Utilizes the latest CHGNet pre-trained transformer model for accurate universal interatomic potential predictions.
*   **One-Click Relaxation**: Automatically optimizes atomic positions and lattice parameters to find the ground-state structure.
*   **Property Prediction**: Instantly calculates:
    *   **Total Energy** (eV/atom)
    *   **Magnetic Moments** (magmom)
    *   **Site-Specific Energies**
*   **Auto Mode Optimization**: An advanced AI-driven feature that iteratively substitutes or deletes atoms to discover the most energetically favorable configuration.
*   **Real-time Feedback**: Watch energy minimization progress live via dynamic sparkline charts.

### 📊 Comprehensive Analysis Tools
*   **Local Analytics Dashboard**: A built-in SQLite database tracks your usage history and calculation statistics. **Data privacy is paramount**: all analytics are stored locally on your machine and are never uploaded to the cloud.
*   **Detailed Metrics**: Inspect precise lattice parameters (a, b, c, α, β, γ), stress tensors, and atomic forces for every step of the relaxation.
*   **Full Data Export**: Download a comprehensive ZIP archive containing:
    *   The final relaxed structure (CIF)
    *   Property data (CSV/TXT)
    *   Full optimization trajectory logs

## Directory Structure

```
CrystalNexus/
├── main.py                 # Core FastAPI backend application
├── start_crystalnexus.py   # robust server startup script with auto-recovery and health monitoring
├── analytics_db.py         # Local analytics database manager (SQLite)
├── sample_cif/            # Curated library of sample crystal structures (Oxides, Metals, etc.)
├── templates/             # Jinja2 HTML templates for the frontend
├── static/               # Static assets (CSS, JS, images)
├── uploads/              # Temporary directory for user uploads (auto-cleaned)
└── requirements.txt        # Detailed Python dependencies list
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

*   **What happens next?**
    *   The script checks if port `8080` is free.
    *   It starts the FastAPI backend server.
    *   It monitors the server health continuously.
    *   Once ready, it will display: `CrystalNexus is ready! URL: http://localhost:8080`

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
*   **Static Calculation**: Click **"Analyze Structure"** (Default). This calculates the energy of the *current* geometry without moving atoms.
*   **Relaxation**: Check the **"Relax Structure"** box before analyzing. This will optimize the geometry. The 3D view will update to show the new, relaxed structure.

#### 4. Advanced Editing
*   **Supercell**: Open the **"Structure Operations"** menu. Enter dimensions (e.g., 2 2 2) and click **"Create Supercell"**.
*   **Auto Mode**: Select "Auto Mode" to let AI iteratively improve your structure by testing substitutions (e.g., substituting Ti with Zr).

#### 5. Exporting Results
*   After an analysis is complete, click the **"Detailed Analysis"** button.
*   In the modal, click **"Download All Data (ZIP)"** to save your work.

---

## Troubleshooting

### Q: "Backend failed to start" or "Port 8080 is in use"
*   **Solution**: The `start_crystalnexus.py` script automatically detects and stops any existing process using port 8080. Simply running `python start_crystalnexus.py` again should resolve the issue. Manual intervention is only necessary in rare cases where the script lacks permission to terminate the process.

### Q: "Module not found: chgnet"
*   **Correction**: Ensure you activated your virtual environment (`venv`) before running the server. Re-run `pip install -r requirements.txt`.

### Q: "DLL load failed" on Windows
*   **Correction**: This usually means a missing system dependency. Install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

### Q: Performance is slow
*   **Correction**: Computing relaxation for large supercells (100+ atoms) can be slow because CrystalNexus is configured to run on **CPU only** to ensure broad compatibility and stability. Please be patient when analyzing complex structures or large supercells.

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
