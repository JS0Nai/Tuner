#!/usr/bin/env python3
"""
LLM Fine-tuning Pipeline
-----------------------
This script orchestrates the complete pipeline for preparing, cleaning, and
optimizing content for language model fine-tuning.
"""

import os
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
import json
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


class FineTuningPipeline:
    """Orchestrates the complete fine-tuning preparation pipeline."""
    
    def __init__(self, base_dir=None, sources=None, source_file=None, source_dir=None,
                 instruction="Continue writing in the style of the author:",
                 model="gpt-3.5-turbo"):
        # Set base directory for the pipeline
        self.base_dir = Path(base_dir) if base_dir else Path("finetuning")
        self.base_dir.mkdir(exist_ok=True)
        # Instruction/prompt to pass into optimizer, extractor, and creator
        self.instruction = instruction
        # LLM model for command extraction
        self.model = model
        # Directories for each stage
        
        # Define directories for each stage
        self.raw_dir = self.base_dir / "raw"
        self.cleaned_dir = self.base_dir / "cleaned"
        self.optimized_dir = self.base_dir / "optimized"
        self.refined_dir = self.base_dir / "refined"
        self.final_dir = self.base_dir / "final"
        
        # Ensure all directories exist
        for directory in [self.raw_dir, self.cleaned_dir, self.optimized_dir, self.refined_dir, self.final_dir]:
            directory.mkdir(exist_ok=True)
        
        # Store source arguments
        self.sources = sources
        self.source_file = source_file
        self.source_dir = source_dir
        
        # Pipeline metrics
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "stages": {},
            "total_duration_seconds": 0
        }
    
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        # Python packages to check
        required_packages = [
            "beautifulsoup4", "trafilatura", "nltk", "spacy",
            "markdown", "pypandoc", "pdfplumber"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing Python packages: {', '.join(missing_packages)}")
            logger.info("Installing missing packages...")
            
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade"] + missing_packages, check=True)
                logger.info("All missing packages installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e}")
                return False
        
        return True
    
    def run_content_extraction(self):
        """Run the content extraction stage."""
        logger.info("Stage 1: Content Extraction")
        stage_start = time.time()
        
        cmd = [sys.executable, "content_extractor.py", "--output", str(self.raw_dir)]
        
        # Add source arguments
        if self.sources:
            cmd.extend(["--sources"] + self.sources)
        elif self.source_file:
            cmd.extend(["--file", self.source_file])
        elif self.source_dir:
            cmd.extend(["--dir", self.source_dir])
        else:
            logger.error("No sources specified for content extraction")
            return False
        
        # Run extraction script
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            
            # Check if files were generated
            raw_files = list(self.raw_dir.glob("*.json"))
            logger.info(f"Extraction complete. {len(raw_files)} files generated.")
            
            # Record metrics
            duration = time.time() - stage_start
            self.metrics["stages"]["extraction"] = {
                "files_generated": len(raw_files),
                "duration_seconds": duration
            }
            
            return len(raw_files) > 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Content extraction failed: {e}")
            return False
    
    def run_text_cleaning(self):
        """Run the text cleaning stage."""
        logger.info("Stage 2: Text Cleaning")
        stage_start = time.time()
        
        cmd = [
            sys.executable, "text_cleaner.py",
            "--input", str(self.raw_dir),
            "--output", str(self.cleaned_dir)
        ]
        
        # Run cleaning script
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            
            # Check if files were generated
            cleaned_files = list(self.cleaned_dir.glob("*.json"))
            logger.info(f"Cleaning complete. {len(cleaned_files)} files generated.")
            
            # Record metrics
            duration = time.time() - stage_start
            self.metrics["stages"]["cleaning"] = {
                "files_generated": len(cleaned_files),
                "duration_seconds": duration
            }
            
            return len(cleaned_files) > 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Text cleaning failed: {e}")
            return False
    
    def run_content_optimization(self):
        """Run the content optimization stage."""
        logger.info("Stage 3: Content Optimization")
        stage_start = time.time()
        
        cmd = [
            sys.executable, "content_optimizer.py",
            "--input", str(self.cleaned_dir),
            "--output", str(self.optimized_dir),
            "--instruction", self.instruction
        ]
        
        # Run optimization script
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            
            # Check if files were generated
            optimized_files = list(self.optimized_dir.glob("*.json"))
            logger.info(f"Optimization complete. {len(optimized_files)} files generated.")
            
            # Record metrics
            duration = time.time() - stage_start
            self.metrics["stages"]["optimization"] = {
                "files_generated": len(optimized_files),
                "duration_seconds": duration
            }
            
            return len(optimized_files) > 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Content optimization failed: {e}")
            return False
    
    def run_command_extraction(self):
        """Run the AI-driven command extraction/refinement stage."""
        logger.info("Stage 3.5: Command extraction/refinement")
        stage_start = time.time()
        cmd = [
            sys.executable, "command_extractor.py",
            "--input", str(self.optimized_dir),
            "--output", str(self.refined_dir),
            "--instruction", self.instruction,
            "--model", self.model
        ]
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            files = list(self.refined_dir.glob("*.json"))
            duration = time.time() - stage_start
            self.metrics["stages"]["refinement"] = {
                "files_generated": len(files),
                "duration_seconds": duration
            }
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command extraction failed: {e}")
            return False
    
    def run_dataset_creation(self, val_ratio=0.1):
        """Run the dataset creation stage."""
        logger.info("Stage 4: Final Dataset Creation")
        stage_start = time.time()
        
        # Use refined content if available
        input_dir = self.refined_dir if any(self.refined_dir.glob("*.json")) else self.optimized_dir
        cmd = [
            sys.executable, "dataset_creator.py",
            "--input", str(input_dir),
            "--output", str(self.final_dir),
            "--val-ratio", str(val_ratio),
            "--instruction", self.instruction
        ]
        
        # Run dataset creation script
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            
            # Check if directories were generated
            format_dirs = [d for d in self.final_dir.iterdir() if d.is_dir()]
            
            # Check if metadata file was generated
            metadata_file = self.final_dir / "dataset_metadata.json"
            
            logger.info(f"Dataset creation complete. {len(format_dirs)} format directories generated.")
            
            # Record metrics
            duration = time.time() - stage_start
            self.metrics["stages"]["dataset_creation"] = {
                "formats_generated": len(format_dirs),
                "duration_seconds": duration
            }
            
            # If metadata file exists, load and add to metrics
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        dataset_metadata = json.load(f)
                    
                    self.metrics["dataset_summary"] = {
                        "total_examples": dataset_metadata.get("total_examples", 0),
                        "total_segments": dataset_metadata.get("total_segments", 0),
                        "train_examples": dataset_metadata.get("split", {}).get("train", 0),
                        "validation_examples": dataset_metadata.get("split", {}).get("validation", 0)
                    }
                except Exception as e:
                    logger.error(f"Error reading dataset metadata: {e}")
            
            return len(format_dirs) > 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Dataset creation failed: {e}")
            return False
    
    def save_pipeline_metrics(self):
        """Save metrics for the entire pipeline."""
        end_time = datetime.now()
        self.metrics["end_time"] = end_time.isoformat()
        
        # Calculate total duration
        start_time = datetime.fromisoformat(self.metrics["start_time"])
        total_duration = (end_time - start_time).total_seconds()
        self.metrics["total_duration_seconds"] = total_duration
        
        # Save metrics to file
        metrics_path = self.base_dir / "pipeline_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Pipeline metrics saved to {metrics_path}")
    
    def run_pipeline(self, val_ratio=0.1):
        """Run the complete pipeline."""
        logger.info("Starting LLM fine-tuning preparation pipeline")
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("Dependency check failed. Aborting pipeline.")
            return False
        
        # Stage 1: Content Extraction
        if not self.run_content_extraction():
            logger.error("Content extraction stage failed. Aborting pipeline.")
            return False
        
        # Stage 2: Text Cleaning
        if not self.run_text_cleaning():
            logger.error("Text cleaning stage failed. Aborting pipeline.")
            return False
        
        # Stage 3: Content Optimization
        if not self.run_content_optimization():
            logger.error("Content optimization stage failed. Aborting pipeline.")
            return False
        # Stage 3.5: AI-driven command extraction/refinement
        if not self.run_command_extraction():
            logger.error("Command extraction stage failed. Aborting pipeline.")
            return False
        
        # Stage 4: Final Dataset Creation
        if not self.run_dataset_creation(val_ratio):
            logger.error("Dataset creation stage failed. Aborting pipeline.")
            return False
        
        # Save pipeline metrics
        self.save_pipeline_metrics()
        
        logger.info("Pipeline completed successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run the complete LLM fine-tuning preparation pipeline")
    
    # Base directory
    parser.add_argument("--base-dir", default="finetuning", help="Base directory for the pipeline")
    
    # Source input options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--sources", "-s", nargs="+", help="List of sources (URLs or file paths)")
    source_group.add_argument("--file", "-f", help="File containing sources (one per line)")
    source_group.add_argument("--dir", "-d", help="Directory to recursively scan for compatible files")
    
    # Validation ratio
    parser.add_argument("--val-ratio", "-v", type=float, default=0.1, help="Validation set ratio")
    # Instruction/prompt for command extraction
    parser.add_argument("--instruction", "-t", default="Continue writing in the style of the author:", help="Instruction for command extraction and refinement")
    # LLM model for API calls
    parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="OpenAI model to use for command extraction")
    
    args = parser.parse_args()
    
    pipeline = FineTuningPipeline(
        base_dir=args.base_dir,
        sources=args.sources,
        source_file=args.file,
        source_dir=args.dir,
        instruction=args.instruction,
        model=args.model
    )
    
    success = pipeline.run_pipeline(val_ratio=args.val_ratio)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()