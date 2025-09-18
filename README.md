# CrystalNexus

A comprehensive web-based crystal structure analysis and machine learning application for materials science research. CrystalNexus integrates CHGNet for structure relaxation, energy prediction, and detailed property analysis with an intuitive interface and powerful data export capabilities.

## Features

### 🔬 **Crystal Structure Analysis**
- **CIF File Analysis**: Load and analyze crystallographic information files (CIF)
- **Interactive 3D Visualization**: Real-time 3D crystal structure visualization with 3Dmol.js
- **Supercell Generation**: Create supercells with custom dimensions
- **Structure Modification**: 
  - **Manual Mode**: Select specific atoms for substitution or deletion
  - **Auto Mode (AI-Optimized)**: CHGNet-powered optimal atom selection for substitution/deletion
    - Automatically tests all atoms of selected element type
    - Uses CHGNet machine learning energy calculations to find the most energy-stable configuration
    - Real-time progress display showing current best energy and selected atom
    - Intelligent optimization algorithms for materials property enhancement
    - Supports both element substitution (e.g., Ca → Ti) and atom deletion
    - Cancellable operations with progress tracking
  - Substitute atoms with CHGnet-supported elements (86 elements available)
  - Delete atoms from crystal structures
  - Real-time formula and density updates

### 🤖 **Machine Learning Integration**
- **CHGNet Integration**: Advanced crystal graph neural network for materials property prediction
- **Structure Relaxation**: Automatic atomic position and unit cell optimization
- **Energy Prediction**: Total energy, energy per atom, and site-specific energies
- **Magnetic Properties**: Magnetic moment calculation for each atom
- **Force Analysis**: Atomic forces and stress tensor calculation

### 📊 **Comprehensive Analysis**
- **Detailed Analysis Modal**: In-depth property analysis with multiple data views
- **Energy & Magnetic Summary**: Quick overview of key properties
- **Step-by-Step Trajectory**: Relaxation progress tracking with convergence analysis
- **Final Atomic Properties**: Individual atomic site energies, magnetic moments, and forces
- **Stress Tensor Analysis**: Complete stress state evaluation

### 💾 **Data Export & Download**
- **ZIP Archive Generation**: Comprehensive analysis data export
- **Intelligent Naming**: Formula-based filenames with timestamps
- **Multiple File Formats**:
  - Crystal structure (CIF with P1 space group)
  - Crystal information (CSV)
  - Energy and magnetic properties (TXT)
  - Final atomic properties (CSV)
  - Stress tensor components (CSV)
  - Step-by-step trajectory data (CSV)

### 🎨 **Professional Interface**
- **Responsive Design**: Clean, modern UI optimized for research workflows
- **Branded Interface**: Custom logo and favicon for professional appearance
- **One-Click Copy**: Copy crystal information, energy data, and analysis results to clipboard
- **Real-time Updates**: Live property updates during modifications
- **Toast Notifications**: User-friendly feedback system
- **Progress Tracking**: Visual indicators for long-running calculations
- **Comprehensive User Guide**: Built-in documentation accessible from the interface

## Directory Structure

```
CrystalNexus/
├── main.py                 # FastAPI backend application
├── start_crystalnexus.py   # Smart startup script with health monitoring
├── requirements.txt        # Python dependencies (2025 latest versions)
├── .gitignore             # Git ignore configuration
├── README.md              # Project documentation
├── sample_cif/            # Sample crystal structure files (organized by category)
│   ├── Gases/            # Gas phase structures
│   │   ├── CO2(gas).cif   # Carbon dioxide
│   │   ├── H2(gas).cif    # Hydrogen gas
│   │   └── H2O(gas).cif   # Water vapor
│   ├── Metals/           # Metallic structures
│   │   ├── Cu.cif         # Copper
│   │   ├── Nd2Fe14B.cif   # Neodymium iron boron (permanent magnet)
│   │   ├── Ni.cif         # Nickel
│   │   └── Si.cif         # Silicon
│   └── Oxides/           # Oxide ceramics
│       ├── BaTiO3(cubic).cif      # Barium titanate (cubic perovskite)
│       ├── BaTiO3(tetragonal).cif # Barium titanate (tetragonal)
│       ├── CaZrO3.cif     # Calcium zirconate
│       ├── FeBiO3.cif     # Iron bismuth oxide
│       ├── SrFe12O19.cif  # Strontium ferrite (hexaferrite)
│       └── ZrO2.cif       # Zirconia (ceramic)
├── templates/             # HTML templates
│   └── index.html        # Main application interface
├── static/               # Static web assets
│   ├── css/              # Stylesheets
│   ├── js/               # JavaScript files
│   ├── favicon.ico       # Browser favicon
│   ├── crystalnexus_whitelogo.png  # Application logo
│   └── user_guide.html   # Comprehensive user documentation
```

## Installation & Setup

### Prerequisites
- **Python 3.8 or higher**
- **pip package manager**
- **Git** (for cloning the repository)
- **Windows Only**: Microsoft Visual C++ Redistributable (required for pymatgen and CHGNet)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/ToshihiroIguchi/CrystalNexus.git
cd CrystalNexus
```

2. **Create and activate a virtual environment (recommended):**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
# Install latest packages (2025 versions)
pip install -r requirements.txt
```
**Important Notes:**
- Installation may take 10-15 minutes due to PyTorch 2.8.0 and CHGNet 0.4.0 dependencies
- **Windows users**: Visual C++ Build Tools 2022 required for CHGNet 0.4.0 (see troubleshooting section)
- All packages updated to 2025 latest versions for improved performance and compatibility
- If installation fails, try: `pip install --upgrade pip` first

4. **Start the application:**

**Windows (Recommended):**
```cmd
# Command Prompt
venv\Scripts\activate
python start_crystalnexus.py
```

**PowerShell:**
```powershell
venv_latest\Scripts\Activate.ps1
python start_crystalnexus.py
```

**macOS/Linux:**
```bash
source venv_latest/bin/activate
python start_crystalnexus.py
```

5. **Access the application:**
- Open your web browser
- Navigate to: **http://localhost:8080** (default port, configurable via CRYSTALNEXUS_PORT environment variable)

### **start_crystalnexus.py Features:**
The intelligent startup script provides comprehensive server management:
- **Automated Health Check**: Detects existing CrystalNexus instances and prevents conflicts
- **Smart Process Management**: Gracefully stops existing servers and starts fresh instances
- **Environment-Aware Configuration**: Inherits all settings from main.py (HOST, PORT, DEBUG)
- **Real-time Startup Monitoring**: Visual progress indicators and server health status
- **Advanced Error Recovery**: Automatic server restart on failure with detailed error reporting
- **Signal Handling**: Clean shutdown with Ctrl+C support and proper process cleanup
- **Cross-platform Compatibility**: Native support for Windows, macOS, and Linux
- **Background Health Monitoring**: Continuous server health verification during operation
- **Port Conflict Resolution**: Automatically handles port 8080 conflicts and process cleanup
- **Debug Mode Support**: Conditional auto-reload functionality in development environments

### Manual Startup (Alternative)

If you prefer to start the backend manually:

```bash
# Start the FastAPI backend
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### Troubleshooting

**Port Already in Use:**
```bash
# Check what's using port 8080
lsof -i :8080  # macOS/Linux
netstat -ano | findstr :8080  # Windows

# Kill the process if needed
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

**Dependencies Issues:**
```bash
# Update pip and try again
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**CHGNet Loading Issues:**
- First run may take longer as CHGNet downloads model weights
- Ensure stable internet connection during initial setup

**Windows-Specific Issues:**

**Missing Visual C++ Build Tools (CHGNet 0.4.0 requirement):**

CrystalNexus now uses CHGNet 0.4.0 which requires compilation on Windows. You may encounter:
```bash
error: Microsoft Visual C++ 14.0 or greater is required. 
Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**Solution: Install Visual Studio Build Tools 2022 (Required for CHGNet 0.4.0)**
1. **Download Visual Studio Build Tools:**
   - Go to: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Click "Download Build Tools for Visual Studio 2022"
   - Run the downloaded installer (`vs_buildtools.exe`)

2. **Install Required Components:**
   - Launch the installer
   - Select **"C++ build tools"** workload
   - Ensure these components are selected:
     - ✅ **MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)**
     - ✅ **Windows 11 SDK (Latest version)**
     - ✅ **CMake tools for Visual Studio**
   - Click **"Install"**
   - **Restart your computer** after installation

3. **Verify Installation:**
   ```cmd
   # Open new Command Prompt and test
   cl
   # Should show: Microsoft (R) C/C++ Optimizing Compiler Version XX.XX.XXXXX
   ```

4. **Install CrystalNexus Dependencies:**
   ```cmd
   # Now install the requirements with CHGNet 0.4.0 support
   pip install -r requirements.txt
   ```

**Complete Installation Steps with Build Tools:**
```cmd
# 1. Install Visual Studio Build Tools (see steps above)
# 2. Restart computer
# 3. Open new Command Prompt
# 4. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 5. Install requirements with CHGNet 0.4.0 and 2025 latest packages
pip install -r requirements.txt
```

**Legacy Compatibility Issues:**
If you encounter issues with the latest versions, ensure you're using Python 3.8+ and have the latest pip version installed.

**Permission Issues:**
```bash
# Run Command Prompt as Administrator if encountering file access errors
# Or ensure Python has write permissions to installation directory
```

## Usage Guide

### 1. **Loading Crystal Structures**
- Click "Select CIF File" to open the file selection modal
- Choose from 13 sample structures organized by category (Gases, Metals, Oxides) or upload your own CIF file
- Supported formats: Standard CIF files with crystallographic data

### 2. **Creating Supercells**
- Specify supercell dimensions (e.g., 2×2×2)
- View real-time updates of atom count and formula
- 3D visualization automatically updates to show expanded structure

### 3. **Structure Analysis with CHGNet**
- Click "Run CHGNet Analysis" to perform ML-based analysis
- Analysis includes:
  - Structure relaxation and optimization
  - Energy and force calculations
  - Magnetic property prediction
  - Stress tensor evaluation

### 4. **Detailed Analysis**
- Click "📊 Detailed Analysis" after running CHGNet analysis
- Explore comprehensive data in organized tabs:
  - **Crystal Info**: Lattice parameters, volume, density
  - **Energy Details**: Total energy, per-atom energies, convergence data
  - **Final Atomic Properties**: Site-specific energies, magnetic moments, forces
  - **Stress Details**: Complete stress tensor components
  - **Trajectory**: Step-by-step optimization progress

### 5. **Structure Modification**
- **Manual Mode**:
  - **Element Substitution**: Replace atoms with CHGNet-supported elements (86 available)
  - **Atom Deletion**: Remove specific atoms from the structure
  - **Direct Selection**: Choose specific atoms from dropdown menu
  - **Immediate Updates**: Real-time formula and property calculations

- **Auto Mode (AI-Optimized)**:
  - **Intelligent Atom Testing**: Systematically evaluates all atoms of selected element type
  - **CHGNet Energy Optimization**: Machine learning-based energy calculations for each configuration
  - **Automatic Best Selection**: Chooses the most thermodynamically stable option
  - **Real-time Progress Display**: Live updates showing current best energy and selected atom
  - **Cancellable Operations**: Stop long-running optimizations with preserved results
  - **Smart Algorithm**: Efficient exploration of configuration space for optimal materials properties

- **Universal Features**:
  - **Real-time Updates**: Instant formula and density recalculation
  - **Structure Reset**: One-click restoration to original configuration
  - **Visual Feedback**: 3D structure updates with modification highlights

### 6. **Data Export**
- Click "💾 Download Analysis" to generate comprehensive ZIP archive
- Files included:
  - `structure_P1.cif`: Relaxed crystal structure (P1 space group)
  - `crystal_information.csv`: Lattice parameters and properties
  - `energy_magnetic.txt`: Energy and magnetic properties summary
  - `final_atomic_properties.csv`: Atomic-level properties
  - `stress_tensor.csv`: Stress tensor components
  - `trajectory_data.csv`: Optimization trajectory
- Filename format: `CrystalNexus_{Formula}_{NumAtoms}atoms_{Timestamp}.zip`

## Sample Crystal Structures

CrystalNexus includes 13 carefully selected sample structures organized by category:

### Gas Phase Structures
| Structure | Formula | Applications |
|-----------|---------|-------------|
| **CO2(gas)** | CO₂ | Carbon capture, atmospheric studies |
| **H2(gas)** | H₂ | Hydrogen fuel, energy storage |
| **H2O(gas)** | H₂O | Water vapor, humidity studies |

### Metallic Structures
| Structure | Formula | Type | Applications |
|-----------|---------|------|-------------|
| **Cu** | Cu | Metal | Electrical conductors, electronics |
| **Nd2Fe14B** | Nd₂Fe₁₄B | Intermetallic | Permanent magnets, motors |
| **Ni** | Ni | Metal | Catalysts, alloys |
| **Si** | Si | Semiconductor | Electronics, solar cells |

### Oxide Ceramics
| Structure | Formula | Type | Applications |
|-----------|---------|------|-------------|
| **BaTiO3(cubic)** | BaTiO₃ | Perovskite | Ferroelectric ceramics, capacitors |
| **BaTiO3(tetragonal)** | BaTiO₃ | Perovskite | Ferroelectric materials |
| **CaZrO3** | CaZrO₃ | Perovskite | High-temperature ceramics |
| **FeBiO3** | FeBiO₃ | Perovskite | Multiferroic materials |
| **SrFe12O19** | SrFe₁₂O₁₉ | Hexaferrite | Magnetic recording, microwave devices |
| **ZrO2** | ZrO₂ | Oxide ceramic | Thermal barrier coatings, fuel cells |

## API Documentation

### Core Endpoints
- `GET /` - Main application interface
- `GET /health` - Application health check
- `GET /api/sample-cif-files` - List available sample structures
- `GET /api/chgnet-elements` - CHGNet supported elements (86 total)

### Analysis Endpoints
- `POST /api/analyze-cif-sample` - Analyze sample CIF file
- `POST /api/analyze-cif-upload` - Analyze uploaded CIF file
- `POST /api/create-supercell` - Generate supercell structure
- `POST /api/chgnet-predict` - Run CHGNet energy prediction
- `POST /api/chgnet-relax` - Run CHGNet structure relaxation
- `POST /api/generate-relaxed-structure-cif` - Get relaxed structure
- `POST /api/generate-modified-structure-cif` - Get modified structure

### Structure Modification
- `POST /api/get-element-labels` - Get atom labels for operations
- `POST /api/substitute-element` - Substitute atom elements
- `POST /api/delete-atom` - Remove atoms from structure

## Development

### Running Tests
```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run basic application tests
python -c "import main; print('Application imports successfully')"

# Test CHGNet integration
python -c "from chgnet.model import CHGNet; CHGNet.load(); print('CHGNet loaded successfully')"

# Verify server startup
python start_crystalnexus.py  # Should start without errors
```

### Development Mode
```bash
# Start with auto-reload for development
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### Code Quality
```bash
# Install development dependencies
pip install black flake8 isort

# Format code
black main.py
isort main.py

# Lint code
flake8 main.py
```

## Dependencies

### Core Dependencies (2025 Latest Versions)
- **FastAPI 0.115.5** - Modern web framework for building APIs  
- **uvicorn 0.35.0** - High-performance ASGI web server
- **pymatgen 2025.6.14** - Advanced materials analysis library
- **chgnet 0.4.0** - Latest crystal graph neural network with improved accuracy
- **torch 2.8.0** - PyTorch machine learning framework

### Supporting Libraries
- **python-multipart 0.0.12** - File upload support
- **jinja2 3.1.6** - Template engine
- **aiofiles 24.1.0** - Async file operations
- **numpy** - Numerical computing (>=1.26.0, <2.4.0)
- **scipy** - Scientific computing (>=1.12.0)
- **requests** - HTTP library for health checks (>=2.32.0)

### Frontend Libraries (CDN)
- **3Dmol.js** - Molecular visualization
- **JSZip** - ZIP file generation

## Performance Notes

- **First Run**: CHGNet model download may take 2-3 minutes
- **Analysis Time**: Typical CHGNet analysis takes 10-30 seconds depending on structure size
- **Memory Usage**: ~2-4 GB RAM recommended for typical structures
- **Browser Compatibility**: Modern browsers with WebGL support required for 3D visualization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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

## Acknowledgments

- **CHGNet Team** - For the excellent crystal graph neural network
- **Materials Project** - For pymatgen library
- **3Dmol.js Team** - For molecular visualization capabilities

## License

MIT License

Copyright (c) 2025 Toshihiro Iguchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
