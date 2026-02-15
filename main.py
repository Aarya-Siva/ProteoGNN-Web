"""
ProteoGNN Web Application - Protein Misfolding Prediction
A Flask-based web interface for predicting residue-level misfolding propensity
in amyloidogenic proteins using Graph Neural Networks.
"""

import os
import json
import uuid
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile

# Import our modules
from model import ProteoGNN, create_model
from graph_builder import ProteinGraphBuilder
from features import NodeFeatureExtractor, EdgeFeatureExtractor

import torch
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Global model instance
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration matching trained model
MODEL_CONFIG = {
    'node_input_dim': 46,  # Feature dimension from NodeFeatureExtractor
    'hidden_dim': 128,
    'num_layers': 4,
    'layer_type': 'graphconv',
    'heads': 4,
    'dropout': 0.1,
    'edge_dim': 13,  # Edge feature dimension
    'use_edge_features': True,
    'residual': True,
}


def load_model():
    """Load the trained model or create a demo model."""
    global MODEL

    if MODEL is not None:
        return MODEL

    logger.info("Creating ProteoGNN model...")
    MODEL = create_model(MODEL_CONFIG)
    MODEL.to(DEVICE)
    MODEL.eval()

    # Try to load pretrained weights if available
    checkpoint_path = Path("checkpoints/best.pt")
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            MODEL.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded pretrained model weights")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
            logger.info("Using randomly initialized model (demo mode)")
    else:
        logger.info("No checkpoint found - using demo model with random weights")

    return MODEL


def process_pdb_content(pdb_content: str, chain_id: str = "A") -> dict:
    """
    Process PDB content and run prediction.

    Args:
        pdb_content: Raw PDB file content
        chain_id: Chain to analyze

    Returns:
        Dictionary with prediction results
    """
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_path = f.name

    try:
        # Build protein graph
        builder = ProteinGraphBuilder()
        graph = builder.build_graph(temp_path, chain_id=chain_id)

        # Load model and predict
        model = load_model()

        with torch.no_grad():
            graph.x = graph.x.to(DEVICE)
            graph.edge_index = graph.edge_index.to(DEVICE)
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                graph.edge_attr = graph.edge_attr.to(DEVICE)

            probs = model.predict_proba(graph).cpu().numpy()

        # Build results
        results = {
            'success': True,
            'structure_id': graph.structure_id,
            'chain_id': graph.chain_id,
            'num_residues': graph.num_nodes,
            'num_edges': graph.edge_index.shape[1],
            'residues': [],
            'coordinates': graph.pos.tolist(),
            'edges': graph.edge_index.t().tolist(),
        }

        for i, (res_id, res_name, prob) in enumerate(zip(
            graph.residue_ids, graph.residue_names, probs
        )):
            risk_level = 'high' if prob > 0.7 else 'medium' if prob > 0.4 else 'low'
            results['residues'].append({
                'index': i,
                'residue_id': res_id,
                'residue_name': res_name,
                'probability': float(prob),
                'prediction': int(prob >= 0.5),
                'risk_level': risk_level,
            })

        # Summary statistics
        probs_array = np.array(probs)
        results['summary'] = {
            'mean_probability': float(probs_array.mean()),
            'max_probability': float(probs_array.max()),
            'high_risk_count': int((probs_array > 0.7).sum()),
            'medium_risk_count': int(((probs_array > 0.4) & (probs_array <= 0.7)).sum()),
            'low_risk_count': int((probs_array <= 0.4).sum()),
        }

        return results

    except Exception as e:
        logger.error(f"Error processing PDB: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for protein misfolding prediction.

    Accepts either:
    - File upload (multipart/form-data with 'file' field)
    - Direct PDB content (JSON with 'pdb_content' field)
    """
    chain_id = request.form.get('chain_id', 'A') or request.json.get('chain_id', 'A') if request.json else 'A'

    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        pdb_content = file.read().decode('utf-8')
        chain_id = request.form.get('chain_id', 'A')

    # Handle JSON body
    elif request.json and 'pdb_content' in request.json:
        pdb_content = request.json['pdb_content']
        chain_id = request.json.get('chain_id', 'A')

    else:
        return jsonify({'success': False, 'error': 'No PDB content provided'}), 400

    # Process and return results
    results = process_pdb_content(pdb_content, chain_id)

    if results['success']:
        return jsonify(results)
    else:
        return jsonify(results), 500


@app.route('/api/fetch_pdb/<pdb_id>')
def fetch_pdb(pdb_id):
    """Fetch a PDB file from RCSB."""
    import urllib.request

    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            pdb_content = response.read().decode('utf-8')
        return jsonify({
            'success': True,
            'pdb_id': pdb_id,
            'content': pdb_content
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Could not fetch PDB {pdb_id}: {str(e)}'
        }), 404


@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """Export prediction results as CSV."""
    data = request.json
    if not data or 'residues' not in data:
        return jsonify({'error': 'No data provided'}), 400

    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Residue_ID', 'Residue_Name', 'Probability', 'Prediction', 'Risk_Level'])

    for res in data['residues']:
        writer.writerow([
            res['residue_id'],
            res['residue_name'],
            f"{res['probability']:.4f}",
            res['prediction'],
            res['risk_level']
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='proteognn_predictions.csv'
    )


@app.route('/api/export/pymol', methods=['POST'])
def export_pymol():
    """Export PyMOL visualization script."""
    data = request.json
    if not data or 'residues' not in data:
        return jsonify({'error': 'No data provided'}), 400

    script = generate_pymol_script(data)

    import io
    return send_file(
        io.BytesIO(script.encode()),
        mimetype='text/plain',
        as_attachment=True,
        download_name='proteognn_visualization.pml'
    )


def generate_pymol_script(data: dict) -> str:
    """Generate a PyMOL visualization script."""
    lines = [
        "# ProteoGNN Misfolding Prediction Visualization",
        "# Generated by ProteoGNN Web App",
        "",
        "# Load your structure first:",
        "# load your_structure.pdb",
        "",
        "# Color settings",
        "bg_color white",
        "set cartoon_fancy_helices, 1",
        "set cartoon_flat_sheets, 1",
        "hide all",
        "show cartoon",
        "",
        "# Color by misfolding probability",
        "# Blue = low risk, White = medium, Red = high risk",
        "set_color low_risk, [0.2, 0.4, 0.8]",
        "set_color medium_risk, [1.0, 1.0, 0.8]",
        "set_color high_risk, [0.9, 0.2, 0.2]",
        "",
    ]

    # Color each residue
    for res in data['residues']:
        prob = res['probability']
        res_id = res['residue_id']

        if prob > 0.7:
            color = "high_risk"
        elif prob > 0.4:
            color = "medium_risk"
        else:
            color = "low_risk"

        lines.append(f"color {color}, resi {res_id}")

    # Show high-risk residues as sticks
    lines.extend([
        "",
        "# Highlight high-risk residues",
    ])

    high_risk = [str(r['residue_id']) for r in data['residues'] if r['probability'] > 0.7]
    if high_risk:
        selection = "+".join(high_risk)
        lines.append(f"show sticks, resi {selection}")
        lines.append(f"color red, resi {selection} and elem C")

    lines.extend([
        "",
        "# Final setup",
        "zoom",
        "ray 1200, 1200",
    ])

    return "\n".join(lines)


@app.route('/api/demo')
def demo_data():
    """Return demo prediction data for testing the UI."""
    # Sample tau protein PHF6 region data
    demo_residues = [
        {'index': 0, 'residue_id': 301, 'residue_name': 'GLY', 'probability': 0.25, 'prediction': 0, 'risk_level': 'low'},
        {'index': 1, 'residue_id': 302, 'residue_name': 'SER', 'probability': 0.32, 'prediction': 0, 'risk_level': 'low'},
        {'index': 2, 'residue_id': 303, 'residue_name': 'PRO', 'probability': 0.28, 'prediction': 0, 'risk_level': 'low'},
        {'index': 3, 'residue_id': 304, 'residue_name': 'GLY', 'probability': 0.35, 'prediction': 0, 'risk_level': 'low'},
        {'index': 4, 'residue_id': 305, 'residue_name': 'THR', 'probability': 0.55, 'prediction': 1, 'risk_level': 'medium'},
        {'index': 5, 'residue_id': 306, 'residue_name': 'VAL', 'probability': 0.89, 'prediction': 1, 'risk_level': 'high'},
        {'index': 6, 'residue_id': 307, 'residue_name': 'GLN', 'probability': 0.85, 'prediction': 1, 'risk_level': 'high'},
        {'index': 7, 'residue_id': 308, 'residue_name': 'ILE', 'probability': 0.92, 'prediction': 1, 'risk_level': 'high'},
        {'index': 8, 'residue_id': 309, 'residue_name': 'VAL', 'probability': 0.88, 'prediction': 1, 'risk_level': 'high'},
        {'index': 9, 'residue_id': 310, 'residue_name': 'TYR', 'probability': 0.82, 'prediction': 1, 'risk_level': 'high'},
        {'index': 10, 'residue_id': 311, 'residue_name': 'LYS', 'probability': 0.78, 'prediction': 1, 'risk_level': 'high'},
        {'index': 11, 'residue_id': 312, 'residue_name': 'PRO', 'probability': 0.45, 'prediction': 0, 'risk_level': 'medium'},
        {'index': 12, 'residue_id': 313, 'residue_name': 'VAL', 'probability': 0.38, 'prediction': 0, 'risk_level': 'low'},
        {'index': 13, 'residue_id': 314, 'residue_name': 'ASP', 'probability': 0.22, 'prediction': 0, 'risk_level': 'low'},
        {'index': 14, 'residue_id': 315, 'residue_name': 'LEU', 'probability': 0.35, 'prediction': 0, 'risk_level': 'low'},
    ]

    probs = [r['probability'] for r in demo_residues]

    return jsonify({
        'success': True,
        'structure_id': 'demo_tau_phf6',
        'chain_id': 'A',
        'num_residues': len(demo_residues),
        'num_edges': 42,
        'residues': demo_residues,
        'summary': {
            'mean_probability': float(np.mean(probs)),
            'max_probability': float(np.max(probs)),
            'high_risk_count': sum(1 for p in probs if p > 0.7),
            'medium_risk_count': sum(1 for p in probs if 0.4 < p <= 0.7),
            'low_risk_count': sum(1 for p in probs if p <= 0.4),
        },
        'is_demo': True
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'device': DEVICE,
        'model_loaded': MODEL is not None
    })


if __name__ == '__main__':
    # Preload model
    load_model()

    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
