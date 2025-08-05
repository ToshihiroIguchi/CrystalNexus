# CrystalNexus

A web-based crystal structure analysis and manipulation tool using FastAPI, pymatgen, and 3Dmol.js.

## Features

- CIF file loading (sample files or user uploads)
- Crystal structure analysis using pymatgen
- Interactive 3D visualization (planned)
- Supercell creation
- Element substitution and deletion (planned)
- CHGNet calculations (planned)

## Directory Structure

```
CrystalNexus/
├── main.py                 # FastAPI backend
├── start_crystalnexus.py   # Startup script
├── requirements.txt        # Python dependencies
├── sample_cif/            # Sample CIF files
├── templates/             # HTML templates
│   └── index.html
├── static/                # Static files (CSS, JS)
├── tests/                 # Test files
│   ├── __init__.py
│   └── test_main.py
└── debug/                 # Debug utilities
    ├── __init__.py
    ├── debug_server.py
    └── test_cif_analysis.py
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start the application:
```bash
python start_crystalnexus.py
```

3. Open your browser and go to: http://localhost:8080

## Usage

1. When you first access the application, a modal window will appear
2. Choose to either:
   - Select a sample CIF file from the dropdown
   - Upload your own CIF file
3. The crystal information will be displayed (formula, space group, lattice parameters, etc.)
4. Specify supercell size in a, b, c directions
5. Click "Confirm" to proceed to the main application

## Development

### Running Tests
```bash
cd tests
python -m pytest test_main.py
```

### Debug Mode
```bash
python debug/debug_server.py
```

### Testing CIF Analysis
```bash
python debug/test_cif_analysis.py
```

## API Endpoints

- `GET /` - Main application page
- `GET /health` - Health check
- `GET /api/sample-cif-files` - Get list of sample CIF files
- `POST /api/analyze-cif-sample` - Analyze sample CIF file
- `POST /api/analyze-cif-upload` - Analyze uploaded CIF file
- `POST /api/create-supercell` - Create supercell structure

## Current Status

✅ **Implemented:**
- FastAPI backend with port 8080
- CIF file loading (sample and upload)
- Pymatgen integration for crystal analysis
- Modal window for file selection
- Supercell size specification
- Backend startup checking
- Test and debug infrastructure

🚧 **In Progress:**
- 3Dmol.js integration for visualization

📋 **Planned:**
- Element substitution and deletion
- CHGNet integration for calculations
- Full 3D structure manipulation
- Advanced crystal structure operations

## Dependencies

- FastAPI 0.104.1
- uvicorn 0.24.0
- pymatgen 2023.10.11
- chgnet 0.3.8
- python-multipart 0.0.6
- jinja2 3.1.2
- aiofiles 23.2.1# CrystalNexus
