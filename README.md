# CrystalNexus

A web-based crystal structure analysis application for materials science research.

## Features

- **CIF File Analysis**: Load and analyze crystallographic information files (CIF)
- **Crystal Structure Visualization**: Interactive 3D visualization of crystal structures  
- **Supercell Generation**: Create supercells with custom dimensions
- **Element Operations**: 
  - Substitute atoms with CHGnet-supported elements (86 elements available)
  - Delete atoms from crystal structures
  - Real-time formula updates
- **Dynamic Modifications**: Progressive structure modifications with reset capability
- **Professional UI**: Clean, responsive interface with toast notifications

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

1. **Load Structure**: Choose from sample CIF files or upload your own
2. **Create Supercell**: Specify dimensions (e.g., 2×2×2) 
3. **Modify Structure**: 
   - Select atoms from dropdown
   - Choose substitution element or deletion
   - Execute operations with real-time updates
4. **Reset**: Restore to original structure anytime

## Sample CIF Files

The application includes several sample crystal structures:
- ZrO2 (Zirconia)
- BaTiO3 (Barium Titanate) 
- Nd2Fe14B (Neodymium Iron Boron)
- SrFe12O19 (Strontium Ferrite)
- C (Carbon/Diamond)

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
- `POST /api/get-element-labels` - Get atom labels for operations
- `GET /api/chgnet-elements` - Get CHGnet supported elements
- `POST /api/recalculate-density` - Recalculate density with pymatgen

## Current Status

✅ **Implemented:**
- FastAPI backend with port 8080
- CIF file loading (sample and upload)
- Pymatgen integration for crystal analysis
- Modal window for file selection
- Supercell size specification
- Element substitution and deletion operations
- CHGnet integration (86 supported elements)
- Dynamic structure modification system
- Pymatgen-based density recalculation
- Reset functionality
- Toast notification system
- Professional UI design
- Complete test suite

📋 **Planned:**
- 3Dmol.js integration for visualization
- Advanced crystal structure operations

## Dependencies

- FastAPI 0.104.1
- uvicorn 0.24.0
- pymatgen 2023.10.11
- chgnet 0.3.8
- python-multipart 0.0.6
- jinja2 3.1.2
- aiofiles 23.2.1

## License

MIT License
