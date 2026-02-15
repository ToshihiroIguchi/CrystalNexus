# CrystalNexus

CrystalNexus is a comprehensive web-based application for crystal structure analysis and materials science research. It integrates the **CHGNet** (Crystal Hamiltonian Graph Neural Network) machine learning model to provide real-time structure relaxation, energy prediction, and property analysis through an intuitive user interface.

## Key Features

### Crystal Structure Analysis
*   **CIF File Support**: Load, analyze, and visualize standard Crystallographic Information Files (CIF)
*   **3D Visualization**: Real-time interactive crystal structure rendering using 3Dmol.js
*   **Supercell Generation**: Create custom supercells with automatic visualization updates
*   **Materials Project Integration**: Integrated search link to the Materials Project database for easy access to crystal structures

### Machine Learning Integration (CHGNet)
*   **Structure Relaxation**: Automatic atomic position and lattice optimization
*   **Property Prediction**: Determine total energy, magmom, and site-specific energies
*   **Auto Mode Optimization**: AI-driven atom substitution and deletion to find the most energetically favorable configurations
*   **Real-time Feedback**: Visual sparkline charts tracking energy changes during auto-optimization

### Comprehensive Analysis Tools
*   **Local Analytics**: Built-in SQLite database tracks usage history and calculation statistics (local only, privacy-focused)
*   **Detailed Metrics**: View lattice parameters, stress tensors, and atomic forces
*   **Data Export**: Download complete analysis results including relaxed structures (CIF), property data (CSV/TXT), and trajectory logs in a single ZIP archive

## Directory Structure

```
CrystalNexus/
├── main.py                 # FastAPI backend application
├── start_crystalnexus.py   # Server startup script with health monitoring
├── analytics_db.py         # Local analytics database manager
├── sample_cif/            # Curated library of sample crystal structures
├── templates/             # HTML templates (Jinja2)
├── static/               # Static assets (CSS, JS, images)
├── uploads/              # Temporary directory for user uploads
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
*   Python 3.8+
*   pip package manager
*   Git

### Quick Start

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ToshihiroIguchi/CrystalNexus.git
    cd CrystalNexus
    ```

2.  **Set up Virtual Environment**
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installation may take a few minutes due to PyTorch and CHGNet dependencies.*

4.  **Run the Application**
    ```bash
    python start_crystalnexus.py
    ```
    Access the application at `http://localhost:8080`.

## API Documentation

CrystalNexus provides a RESTful API powered by FastAPI.

### Core Endpoints
*   `GET /`: Main application interface
*   `GET /health`: System health check
*   `GET /api/sample-cif-files`: List available sample structures

### Analysis & Calculation
*   `POST /api/analyze-cif-upload`: Process uploaded CIF files
*   `POST /api/chgnet-predict`: Run static energy calculation
*   `POST /api/chgnet-relax`: Perform structure relaxation
*   `POST /api/create-supercell`: Generate supercell structures

### Structure Modification
*   `POST /api/apply-atomic-operations`: Apply batch modifications (substitution/deletion)
*   `POST /api/generate-modified-structure-cif`: Export modified structure as CIF

## Technology Stack

*   **Backend**: FastAPI, Uvicorn, Python 3.8+
*   **Machine Learning**: CHGNet, PyTorch, Pymatgen
*   **Frontend**: HTML5, Vanilla JS, Jinja2, 3Dmol.js
*   **Database**: SQLite (for local analytics)

## Contributing

Contributions are welcome. Please ensure to follow the existing code style and submit pull requests for any new features or bug fixes.

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
