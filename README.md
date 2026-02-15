# ProteoGNN Web Application

A beautiful, modern web interface for protein misfolding prediction using Graph Neural Networks.

## Features

- **3D Molecular Visualization**: Interactive 3Dmol.js viewer with prediction coloring
- **Real-time Predictions**: Upload PDB files or fetch from RCSB PDB
- **Risk Analysis**: Per-residue misfolding probability with risk categorization
- **Export Options**: Download results as CSV or PyMOL scripts
- **Modern UI**: Responsive dark theme with smooth animations

## Quick Start on Replit

1. Fork this repl or create a new Python repl
2. Upload all files from the `webapp` folder
3. Click **Run** - the app will start automatically
4. Open the webview to see the application

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Or with gunicorn for production
gunicorn --bind 0.0.0.0:5000 main:app
```

## Usage

### Upload a PDB File
1. Click the upload zone or drag-and-drop a `.pdb` file
2. Select the chain ID (default: A)
3. Click "Analyze Structure"

### Fetch from RCSB PDB
1. Enter a PDB ID (e.g., `5O3L`, `6CU7`)
2. Click "Fetch"
3. Click "Analyze Structure"

### Example Proteins
- **5O3L**: Tau protein (Alzheimer's disease)
- **6CU7**: α-Synuclein (Parkinson's disease)
- **2NAO**: Amyloid-β peptide

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application page |
| `/api/predict` | POST | Run prediction on PDB content |
| `/api/fetch_pdb/<id>` | GET | Fetch PDB from RCSB |
| `/api/demo` | GET | Load demo prediction data |
| `/api/export/csv` | POST | Export results as CSV |
| `/api/export/pymol` | POST | Export PyMOL visualization script |
| `/health` | GET | Health check endpoint |

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla JS with 3Dmol.js
- **ML Framework**: PyTorch
- **Visualization**: 3Dmol.js, Chart.js

## Architecture

```
webapp/
├── main.py              # Flask application & API routes
├── model.py             # ProteoGNN model architecture
├── graph_builder.py     # PDB parsing & graph construction
├── features.py          # Feature extraction utilities
├── templates/
│   └── index.html       # Main frontend (HTML/CSS/JS)
├── requirements.txt     # Python dependencies
├── .replit              # Replit configuration
├── replit.nix           # Nix packages for Replit
└── pyproject.toml       # Project metadata
```

## Risk Levels

| Level | Probability | Description |
|-------|-------------|-------------|
| 🔴 High | >70% | High misfolding propensity |
| 🟡 Medium | 40-70% | Moderate risk |
| 🟢 Low | <40% | Structurally stable |

## Citation

If you use ProteoGNN in your research, please cite:

```
@software{proteognn2024,
  title = {ProteoGNN: Graph Neural Network for Protein Misfolding Prediction},
  year = {2024},
  url = {https://github.com/Aarya-Siva/ProteoGNN-Web}
}
```

## License

MIT License - See LICENSE file for details.
