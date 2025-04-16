#!/usr/bin/env python3
"""
Content Optimizer for LLM Fine-tuning
------------------------------------
This script optimizes cleaned content for language model fine-tuning,
focusing on creating well-structured examples that capture writing style.
"""

import os
import re
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime
import concurrent.futures
import random

# NLP tools
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


class ContentOptimizer:
    """Optimizes content for LLM fine-tuning."""
    
    def __init__(self, input_dir="cleaned", output_dir="optimized", instruction="Continue writing in the style of the author:"):
        """
        instruction: the prompt or instruction to use for each training example.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.instruction = instruction
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters for text segmentation
        self.min_segment_len = 200
        self.max_segment_len = 2000
        self.overlap_size = 50
        
    def segment_text(self, text):
        """
        Segment text into meaningful chunks that preserve context and style.
        Uses natural boundaries like paragraphs and sentences.
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        segments = []
        current_segment = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            para_len = len(para)
            
            # If adding this paragraph exceeds max length, finalize current segment
            if current_len + para_len > self.max_segment_len and current_len > self.min_segment_len:
                segments.append('\n\n'.join(current_segment))
                
                # Keep some overlap for context
                overlap_paras = current_segment[-2:] if len(current_segment) >= 2 else current_segment[-1:]
                current_segment = overlap_paras
                current_len = sum(len(p) for p in overlap_paras)
            
            # Add current paragraph
            current_segment.append(para)
            current_len += para_len
            
            # If we already exceed max length after adding just one paragraph,
            # we need to split by sentences
            if current_len > self.max_segment_len:
                # Join all text so far
                full_text = '\n\n'.join(current_segment)
                # Split into sentences
                sentences = sent_tokenize(full_text)
                
                sent_segments = []
                current_sent_segment = []
                current_sent_len = 0
                
                for sent in sentences:
                    sent_len = len(sent)
                    
                    if current_sent_len + sent_len > self.max_segment_len and current_sent_len > self.min_segment_len:
                        sent_segments.append(' '.join(current_sent_segment))
                        
                        # Keep some sentence overlap
                        overlap_sents = current_sent_segment[-2:] if len(current_sent_segment) >= 2 else current_sent_segment[-1:]
                        current_sent_segment = overlap_sents
                        current_sent_len = sum(len(s) for s in overlap_sents)
                    
                    current_sent_segment.append(sent)
                    current_sent_len += sent_len
                
                # Add final sentence segment if it's not empty
                if current_sent_segment:
                    sent_segments.append(' '.join(current_sent_segment))
                
                # Replace current segment processing with these sentence-based segments
                segments.extend(sent_segments)
                current_segment = []
                current_len = 0
        
        # Add final segment if it's not empty and meets minimum length
        if current_segment and current_len >= self.min_segment_len:
            segments.append('\n\n'.join(current_segment))
        
        return segments
    
    def identify_key_style_markers(self, text):
        """
        Identify key stylistic markers in the text that represent the author's style.
        This helps with creating metadata that can be used during fine-tuning.
        """
        style_markers = {
            "avg_sentence_length": 0,
            "avg_paragraph_length": 0,
            "transition_phrases": [],
            "common_phrases": [],
            "stylistic_elements": []
        }
        
        # Process with spaCy for detailed analysis
        doc = nlp(text[:100000])  # Limit size to avoid memory issues
        
        # Calculate average sentence length
        sentences = list(doc.sents)
        if sentences:
            style_markers["avg_sentence_length"] = sum(len(sent) for sent in sentences) / len(sentences)
        
        # Calculate average paragraph length
        paragraphs = text.split('\n\n')
        if paragraphs:
            style_markers["avg_paragraph_length"] = sum(len(para) for para in paragraphs) / len(paragraphs)
        
        # Find transition phrases
        transition_patterns = [
            r'\b(However|Moreover|Furthermore|In addition|Consequently|As a result|Therefore|Thus|Nevertheless|Nonetheless)\b',
            r'\b(First|Second|Third|Fourth|Finally|Lastly|Primarily|Initially)\b',
            r'\b(For example|For instance|Specifically|To illustrate|In particular)\b'
        ]
        
        for pattern in transition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            style_markers["transition_phrases"].extend([m.lower() for m in matches])
        
        # Count frequency and limit to top 10
        transition_counter = {}
        for phrase in style_markers["transition_phrases"]:
            transition_counter[phrase] = transition_counter.get(phrase, 0) + 1
        
        style_markers["transition_phrases"] = sorted(
            transition_counter.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Find common phrases (n-grams that appear frequently)
        word_tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        
        # Generate 2-3 word phrases
        n_grams = []
        for n in range(2, 4):
            for i in range(len(word_tokens) - n + 1):
                n_gram = ' '.join(word_tokens[i:i+n])
                if len(n_gram) > 5:  # Filter out very short phrases
                    n_grams.append(n_gram)
        
        # Count frequency
        phrase_counter = {}
        for phrase in n_grams:
            phrase_counter[phrase] = phrase_counter.get(phrase, 0) + 1
        
        # Get top 10 phrases that appear at least 3 times
        common_phrases = [(phrase, count) for phrase, count in phrase_counter.items() if count >= 3]
        style_markers["common_phrases"] = sorted(
            common_phrases, 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Identify other stylistic elements
        style_patterns = {
            "questions": r'\?',
            "exclamations": r'!',
            "em_dashes": r'—|-{2}',
            "semicolons": r';',
            "parentheticals": r'\([^)]+\)',
            "quotes": r'"[^"]+"',
            "italics_asterisks": r'\*[^*]+\*',
            "italics_underscores": r'_[^_]+_',
            "bold": r'\*\*[^*]+\*\*',
        }
        
        for style_name, pattern in style_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                style_markers["stylistic_elements"].append({
                    "element": style_name,
                    "count": len(matches),
                    "per_1000_chars": len(matches) / (len(text) / 1000)
                })
        
        return style_markers
    
    def format_for_training(self, segments, metadata=None):
        """
        Format the segments into a suitable format for LLM fine-tuning.
        Different formats may be appropriate depending on the LLM system.
        """
        training_examples = []
        
        for i, segment in enumerate(segments):
            # Create a training example with a simple prompt-completion format
            # This format works well for many LLM fine-tuning approaches
            example = {
                "id": f"example_{i+1}",
                "segment": segment,
                "metadata": metadata or {}
            }
            
            # Add formatting suitable for different LLM systems
            # For a generic prompt-response format:
            example["prompt"] = self.instruction
            example["completion"] = segment
            
            # For a Chat-based format
            inst_clean = self.instruction.rstrip(":").strip()
            sys_msg = f"You are an assistant that {inst_clean.lower()}"
            example["chat_format"] = [
                {"role": "system",    "content": sys_msg},
                {"role": "user",      "content": self.instruction},
                {"role": "assistant", "content": segment}
            ]
            
            training_examples.append(example)
        
        return training_examples
    
    def optimize_content_file(self, file_path):
        """Optimize content from a JSON file for fine-tuning."""
        try:
            logger.info(f"Optimizing content from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content_dict = json.load(f)
            
            # Skip if content is too short
            if "content" not in content_dict or len(content_dict["content"]) < self.min_segment_len:
                logger.warning(f"Content in {file_path} is too short for optimization")
                return False
            
            # Segment the text
            segments = self.segment_text(content_dict["content"])
            logger.info(f"Created {len(segments)} segments from {file_path}")
            
            # Skip if no valid segments were created
            if not segments:
                logger.warning(f"No valid segments created from {file_path}")
                return False
            
            # Identify style markers
            style_markers = self.identify_key_style_markers(content_dict["content"])
            
            # Add metadata
            metadata = {
                "source": content_dict.get("source", ""),
                "title": content_dict.get("title", ""),
                "date": content_dict.get("date", ""),
                "style_markers": style_markers
            }
            
            # Format for training
            training_examples = self.format_for_training(segments, metadata)
            
            # Prepare output
            output = {
                "original_source": content_dict.get("source", ""),
                "title": content_dict.get("title", ""),
                "segments_count": len(segments),
                "style_markers": style_markers,
                "training_examples": training_examples,
                "optimization_timestamp": datetime.now().isoformat()
            }
            
            # Save to output directory
            output_path = self.output_dir / file_path.name
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved optimized content to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error optimizing content from {file_path}: {e}")
            return False
    
    def optimize_directory(self, max_workers=4):
        """Optimize all JSON files in the input directory."""
        json_files = list(self.input_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.input_dir}")
            return 0, 0
        
        logger.info(f"Found {len(json_files)} JSON files to optimize")
        
        successful = 0
        failed = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.optimize_content_file, file_path): file_path for file_path in json_files}
            
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    failed += 1
        
        logger.info(f"Optimization complete. Successful: {successful}, Failed: {failed}")
        return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Optimize content for LLM fine-tuning")
    parser.add_argument("--input", "-i", default="cleaned", help="Input directory containing cleaned content")
    parser.add_argument("--output", "-o", default="optimized", help="Output directory for optimized content")
    parser.add_argument("--instruction", "-t", default="Continue writing in the style of the author:", help="Instruction/prompt for each training example")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--min-segment", type=int, default=200, help="Minimum segment length")
    parser.add_argument("--max-segment", type=int, default=2000, help="Maximum segment length")
    parser.add_argument("--overlap", type=int, default=50, help="Segment overlap size")
    
    args = parser.parse_args()
    
    optimizer = ContentOptimizer(input_dir=args.input, output_dir=args.output, instruction=args.instruction)
    optimizer.min_segment_len = args.min_segment
    optimizer.max_segment_len = args.max_segment
    optimizer.overlap_size = args.overlap
    
    optimizer.optimize_directory(max_workers=args.workers)


if __name__ == "__main__":
    main()