#!/usr/bin/env python3
"""
Web Interface for LLM Fine-tuning Pipeline
----------------------------------------
A simple web interface to manage the fine-tuning preparation process.
"""

import os
import sys
import json
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
import logging

from flask import Flask, render_template, request, jsonify, send_file
import waitress

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Create Flask app
app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Global variables for pipeline state
pipeline_state = {
    "running": False,
    "current_stage": None,
    "progress": 0,
    "logs": [],
    "last_updated": None
}

# Base directory for the pipeline
BASE_DIR = Path("blog_finetuning")
BASE_DIR.mkdir(exist_ok=True)

# Configure Flask app templates
TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)

# Check if we need to create the template file
if not (TEMPLATES_DIR / "index.html").exists():
    logger.info("Creating index.html template")
    with open(TEMPLATES_DIR / "index.html", "w") as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Fine-tuning Pipeline</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
        }
        .container {
            max-width: 1200px;
        }
        .card {
            margin-bottom: 20px;
        }
        #logs {
            height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.9rem;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
        }
        .log-entry {
            margin-bottom: 4px;
        }
        .progress {
            height: 25px;
        }
        #stage-info {
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">LLM Fine-tuning Pipeline</h1>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Pipeline Control</h5>
                    </div>
                    <div class="card-body">
                        <form id="pipeline-form">
                            <div class="mb-3">
                                <label for="sourceType" class="form-label">Source Type</label>
                                <select class="form-select" id="sourceType" required>
                                    <option value="">Select source type...</option>
                                    <option value="urls">List of URLs</option>
                                    <option value="file">Upload Document</option>
                                    <option value="directory">Scan Directory</option>
                                </select>
                            </div>
                            
                            <div class="mb-3 source-input" id="urls-input" style="display: none;">
                                <label for="urls" class="form-label">URLs (one per line)</label>
                                <textarea class="form-control" id="urls" rows="5"></textarea>
                            </div>
                            
                            <div class="mb-3 source-input" id="file-input" style="display: none;">
                                <label for="sourceFile" class="form-label">Upload Document</label>
                                <input type="file" class="form-control" id="sourceFile">
                                <div class="form-text">Upload a document directly (HTML, Markdown, PDF, DOCX)</div>
                            </div>
                            
                            <div class="mb-3 source-input" id="directory-input" style="display: none;">
                                <label for="sourceDir" class="form-label">Directory Path</label>
                                <input type="text" class="form-control" id="sourceDir" placeholder="/path/to/content">
                                <div class="form-text">Directory containing your blog content files</div>
                            </div>
                            
                            <div class="mb-3">
                                <label for="valRatio" class="form-label">Validation Set Ratio</label>
                                <input type="range" class="form-range" id="valRatio" min="0.05" max="0.3" step="0.05" value="0.1">
                                <div class="form-text text-center"><span id="valRatioValue">10</span>% of data used for validation</div>
                            </div>
                            
                            <button type="submit" class="btn btn-primary" id="start-btn">Start Pipeline</button>
                            <button type="button" class="btn btn-danger" id="stop-btn" disabled>Stop Pipeline</button>
                        </form>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Pipeline Status</h5>
                    </div>
                    <div class="card-body">
                        <div id="stage-info">Not running</div>
                        <div class="progress mb-3">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" id="progress-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                        <h6>Recent Logs</h6>
                        <div id="logs"></div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Pipeline Results</h5>
                    </div>
                    <div class="card-body" id="results-container">
                        <p class="text-center text-muted">Pipeline results will appear here after processing</p>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Download Results</h5>
                    </div>
                    <div class="card-body">
                        <div class="list-group" id="download-links">
                            <p class="text-center text-muted">Download links will appear here after processing</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Source type selection
        document.getElementById('sourceType').addEventListener('change', function() {
            document.querySelectorAll('.source-input').forEach(el => el.style.display = 'none');
            const selectedType = this.value;
            if (selectedType === 'urls') {
                document.getElementById('urls-input').style.display = 'block';
            } else if (selectedType === 'file') {
                document.getElementById('file-input').style.display = 'block';
            } else if (selectedType === 'directory') {
                document.getElementById('directory-input').style.display = 'block';
            }
        });
        
        // Validation ratio display
        document.getElementById('valRatio').addEventListener('input', function() {
            document.getElementById('valRatioValue').textContent = Math.round(this.value * 100);
        });
        
        // Pipeline form submission
        document.getElementById('pipeline-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const sourceType = document.getElementById('sourceType').value;
            let formData = new FormData();
            
            formData.append('val_ratio', document.getElementById('valRatio').value);
            formData.append('source_type', sourceType);
            
            if (sourceType === 'urls') {
                formData.append('urls', document.getElementById('urls').value);
            } else if (sourceType === 'file') {
                const file = document.getElementById('sourceFile').files[0];
                if (!file) {
                    alert('Please select a file');
                    return;
                }
                formData.append('source_file', file);
            } else if (sourceType === 'directory') {
                formData.append('source_dir', document.getElementById('sourceDir').value);
            }
            
            try {
                const response = await fetch('/start_pipeline', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('start-btn').disabled = true;
                    document.getElementById('stop-btn').disabled = false;
                    updatePipelineStatus();
                } else {
                    // Show detailed error message
                    console.error('Pipeline start error:', result);
                    alert('Error starting pipeline: ' + result.message);
                    
                    // Update logs with the error
                    const logsContainer = document.getElementById('logs');
                    const logEntry = document.createElement('div');
                    logEntry.className = 'log-entry text-danger';
                    logEntry.textContent = 'Error: ' + result.message;
                    logsContainer.appendChild(logEntry);
                    logsContainer.scrollTop = logsContainer.scrollHeight;
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
        
        // Stop pipeline button
        document.getElementById('stop-btn').addEventListener('click', async function() {
            try {
                const response = await fetch('/stop_pipeline', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert('Pipeline stopping...');
                } else {
                    alert('Error stopping pipeline: ' + result.message);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
        
        // Update pipeline status
        function updatePipelineStatus() {
            fetch('/pipeline_status')
                .then(response => response.json())
                .then(status => {
                    // Update progress bar
                    const progressBar = document.getElementById('progress-bar');
                    progressBar.style.width = status.progress + '%';
                    progressBar.textContent = status.progress + '%';
                    
                    // Update stage info
                    const stageInfo = document.getElementById('stage-info');
                    if (status.running) {
                        stageInfo.textContent = 'Running: ' + (status.current_stage || 'Initializing');
                    } else {
                        stageInfo.textContent = 'Not running';
                        document.getElementById('start-btn').disabled = false;
                        document.getElementById('stop-btn').disabled = true;
                    }
                    
                    // Update logs
                    const logsContainer = document.getElementById('logs');
                    logsContainer.innerHTML = '';
                    status.logs.forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry';
                        logEntry.textContent = log;
                        logsContainer.appendChild(logEntry);
                    });
                    logsContainer.scrollTop = logsContainer.scrollHeight;
                    
                    // If pipeline is still running, schedule another update
                    if (status.running) {
                        setTimeout(updatePipelineStatus, 1000);
                    } else {
                        // Pipeline finished, update results
                        updateResults();
                    }
                })
                .catch(error => {
                    console.error('Error fetching pipeline status:', error);
                    setTimeout(updatePipelineStatus, 5000);
                });
        }
        
        // Update results
        function updateResults() {
            fetch('/pipeline_results')
                .then(response => response.json())
                .then(results => {
                    const resultsContainer = document.getElementById('results-container');
                    
                    if (results.success) {
                        let html = '';
                        
                        if (results.metrics) {
                            html += '<h6>Pipeline Metrics</h6>';
                            html += '<table class="table table-sm">';
                            html += '<tr><td>Total Duration</td><td>' + formatDuration(results.metrics.total_duration_seconds) + '</td></tr>';
                            html += '<tr><td>Total Examples</td><td>' + (results.metrics.dataset_summary?.total_examples || 'N/A') + '</td></tr>';
                            html += '<tr><td>Total Segments</td><td>' + (results.metrics.dataset_summary?.total_segments || 'N/A') + '</td></tr>';
                            html += '<tr><td>Training Examples</td><td>' + (results.metrics.dataset_summary?.train_examples || 'N/A') + '</td></tr>';
                            html += '<tr><td>Validation Examples</td><td>' + (results.metrics.dataset_summary?.validation_examples || 'N/A') + '</td></tr>';
                            html += '</table>';
                            
                            html += '<h6>Stage Durations</h6>';
                            html += '<table class="table table-sm">';
                            for (const [stage, data] of Object.entries(results.metrics.stages)) {
                                html += '<tr><td>' + capitalizeFirstLetter(stage) + '</td><td>' + formatDuration(data.duration_seconds) + '</td></tr>';
                            }
                            html += '</table>';
                        } else {
                            html = '<p class="text-center text-muted">No metrics available</p>';
                        }
                        
                        resultsContainer.innerHTML = html;
                        
                        // Update download links
                        updateDownloadLinks(results.directories);
                    } else {
                        resultsContainer.innerHTML = '<p class="text-center text-danger">' + results.message + '</p>';
                    }
                })
                .catch(error => {
                    console.error('Error fetching pipeline results:', error);
                });
        }
        
        // Update download links
        function updateDownloadLinks(directories) {
            const linksContainer = document.getElementById('download-links');
            
            if (directories && directories.length > 0) {
                let html = '';
                
                directories.forEach(dir => {
                    html += '<a href="/download/' + dir + '" class="list-group-item list-group-item-action">';
                    html += '<div class="d-flex w-100 justify-content-between">';
                    html += '<h6 class="mb-1">' + capitalizeFirstLetter(dir) + ' Format</h6>';
                    html += '<small>Download ZIP</small>';
                    html += '</div>';
                    html += '<p class="mb-1">Fine-tuning dataset in ' + capitalizeFirstLetter(dir) + ' format</p>';
                    html += '</a>';
                });
                
                linksContainer.innerHTML = html;
            } else {
                linksContainer.innerHTML = '<p class="text-center text-muted">No datasets available for download</p>';
            }
        }
        
        // Helper functions
        function formatDuration(seconds) {
            if (seconds === undefined) return 'N/A';
            
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            
            if (mins > 0) {
                return mins + 'm ' + secs + 's';
            } else {
                return secs + 's';
            }
        }
        
        function capitalizeFirstLetter(string) {
            return string.charAt(0).toUpperCase() + string.slice(1);
        }
        
        // Check if pipeline is running on page load
        updatePipelineStatus();
    </script>
</body>
</html>""")
else:
    logger.info("Using existing index.html template")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/start_pipeline', methods=['POST'])
def start_pipeline():
    """Start the pipeline with the given parameters."""
    if pipeline_state["running"]:
        return jsonify({"success": False, "message": "Pipeline is already running"})
    
    try:
        # Get parameters from form
        val_ratio = float(request.form.get('val_ratio', 0.1))
        source_type = request.form.get('source_type')
        
        # Prepare source arguments
        source_args = []
        
        if source_type == 'urls':
            urls = request.form.get('urls', '').strip().split('\n')
            urls = [url.strip() for url in urls if url.strip()]
            
            if not urls:
                return jsonify({"success": False, "message": "No URLs provided"})
            
            # Save URLs to a file
            urls_file = BASE_DIR / "urls.txt"
            with open(urls_file, 'w', encoding='utf-8') as f:
                for url in urls:
                    f.write(url + '\n')
            
            source_args = ["--file", str(urls_file)]
            
        elif source_type == 'file':
            if 'source_file' not in request.files:
                return jsonify({"success": False, "message": "No file provided"})
            
            file = request.files['source_file']
            if file.filename == '':
                return jsonify({"success": False, "message": "No file selected"})
            
            # Log information about the uploaded file for debugging
            logger.info(f"Uploaded file: {file.filename}, content_type: {file.content_type}")
            
            try:
                # Save the uploaded file with its original name in a content directory
                content_dir = BASE_DIR / "content"
                content_dir.mkdir(exist_ok=True)
                
                # Create a safe filename using the original name
                import re
                safe_filename = re.sub(r'[^\w\-\.]', '_', file.filename)
                file_path = content_dir / safe_filename
                
                # Save the file
                file.save(file_path)
                logger.info(f"File saved to {file_path}")
                
                # For documents we should use --sources directly instead of --file
                source_args = ["--sources", str(file_path)]
                
                # Log the approach we're taking
                logger.info(f"Using direct source approach for document: {file_path}")
            except Exception as e:
                logger.error(f"Error saving/processing file: {str(e)}")
                return jsonify({"success": False, "message": f"Error processing file: {str(e)}"})
            
        elif source_type == 'directory':
            source_dir = request.form.get('source_dir', '').strip()
            logger.info(f"Directory input: '{source_dir}'")
            
            if not source_dir:
                return jsonify({"success": False, "message": "No directory provided"})
            
            try:
                source_dir_path = Path(source_dir)
                logger.info(f"Directory path: {source_dir_path} (absolute: {source_dir_path.absolute()})")
                
                if not source_dir_path.exists():
                    logger.error(f"Directory does not exist: {source_dir_path}")
                    return jsonify({"success": False, "message": f"Directory does not exist: {source_dir_path}"})
                if not source_dir_path.is_dir():
                    logger.error(f"Path is not a directory: {source_dir_path}")
                    return jsonify({"success": False, "message": f"Path is not a directory: {source_dir_path}"})
                
                # List the directory contents for debugging
                logger.info(f"Directory contents: {[f.name for f in source_dir_path.iterdir()][:10]}")
                source_args = ["--dir", str(source_dir_path)]
            except Exception as e:
                logger.error(f"Error processing directory path: {str(e)}")
                return jsonify({"success": False, "message": f"Error processing directory: {str(e)}"})
            
        else:
            return jsonify({"success": False, "message": "Invalid source type"})
        
        # Build command
        cmd = [
            sys.executable, "fine_tuning_pipeline.py",
            "--base-dir", str(BASE_DIR),
            "--val-ratio", str(val_ratio)
        ] + source_args
        
        # Log the full command for debugging
        logger.info(f"Command to execute: {' '.join(cmd)}")
        
        # Reset pipeline state
        pipeline_state.update({
            "running": True,
            "current_stage": "Initializing",
            "progress": 0,
            "logs": ["Starting pipeline..."],
            "last_updated": datetime.now().isoformat()
        })
        
        # Start pipeline in a separate thread
        thread = threading.Thread(target=run_pipeline, args=(cmd,))
        thread.daemon = True
        thread.start()
        
        return jsonify({"success": True, "message": "Pipeline started"})
        
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route('/stop_pipeline', methods=['POST'])
def stop_pipeline():
    """Stop the running pipeline."""
    if not pipeline_state["running"]:
        return jsonify({"success": False, "message": "Pipeline is not running"})
    
    try:
        # Mark pipeline as stopped
        pipeline_state["running"] = False
        pipeline_state["logs"].append("Pipeline stop requested...")
        
        return jsonify({"success": True, "message": "Pipeline stop requested"})
        
    except Exception as e:
        logger.error(f"Error stopping pipeline: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route('/pipeline_status')
def get_pipeline_status():
    """Get the current status of the pipeline."""
    return jsonify(pipeline_state)


@app.route('/pipeline_results')
def get_pipeline_results():
    """Get the results of the pipeline."""
    try:
        metrics_path = BASE_DIR / "pipeline_metrics.json"
        
        if metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # Get the directories in the final directory
            final_dir = BASE_DIR / "final"
            directories = []
            
            if final_dir.exists():
                directories = [d.name for d in final_dir.iterdir() if d.is_dir()]
            
            return jsonify({
                "success": True,
                "metrics": metrics,
                "directories": directories
            })
        else:
            return jsonify({
                "success": False,
                "message": "No pipeline results found"
            })
            
    except Exception as e:
        logger.error(f"Error getting pipeline results: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })


@app.route('/download/<directory>')
def download_directory(directory):
    """Download a directory as a ZIP file."""
    try:
        import zipfile
        from io import BytesIO
        
        final_dir = BASE_DIR / "final"
        dir_path = final_dir / directory
        
        if not dir_path.exists() or not dir_path.is_dir():
            return "Directory not found", 404
        
        # Create a ZIP file in memory
        memory_file = BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, final_dir)
                    zipf.write(file_path, arcname)
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{directory}_dataset.zip"
        )
        
    except Exception as e:
        logger.error(f"Error creating download: {e}")
        return "Error creating download", 500


def run_pipeline(cmd):
    """Run the pipeline process and update the pipeline state."""
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stage mapping for progress calculation
        stages = {
            "dependencies": 5,
            "extraction": 25,
            "cleaning": 50,
            "optimization": 75,
            "dataset": 95
        }
        
        # Process output
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            
            if line:
                pipeline_state["logs"].append(line)
                pipeline_state["last_updated"] = datetime.now().isoformat()
                
                # Keep only the last 100 log lines
                if len(pipeline_state["logs"]) > 100:
                    pipeline_state["logs"] = pipeline_state["logs"][-100:]
                
                # Update current stage and progress
                for stage, progress in stages.items():
                    if stage in line.lower():
                        pipeline_state["current_stage"] = stage.capitalize()
                        pipeline_state["progress"] = progress
                        break
                
                logger.info(f"Pipeline output: {line}")
        
        # Process completed
        returncode = process.wait()
        
        if returncode == 0:
            pipeline_state["logs"].append("Pipeline completed successfully")
            pipeline_state["progress"] = 100
            pipeline_state["current_stage"] = "Completed"
        else:
            pipeline_state["logs"].append(f"Pipeline failed with exit code {returncode}")
            pipeline_state["current_stage"] = "Failed"
        
        pipeline_state["running"] = False
        pipeline_state["last_updated"] = datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        pipeline_state["logs"].append(f"Error: {str(e)}")
        pipeline_state["current_stage"] = "Error"
        pipeline_state["running"] = False
        pipeline_state["last_updated"] = datetime.now().isoformat()


if __name__ == "__main__":
    try:
        import webbrowser
        port = 5000
        url = f"http://127.0.0.1:{port}"
        
        # Open web browser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        
        # Start server
        print(f"Starting web interface at {url}")
        waitress.serve(app, host="127.0.0.1", port=port)
        
    except KeyboardInterrupt:
        print("Web interface stopped")
    except Exception as e:
        print(f"Error starting web interface: {e}")
