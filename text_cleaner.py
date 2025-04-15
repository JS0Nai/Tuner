#!/usr/bin/env python3
"""
Text Cleaner for LLM Fine-tuning Preparation
-------------------------------------------
This script cleans and preprocesses extracted content to improve quality
for language model fine-tuning.
"""

import os
import re
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime
import concurrent.futures
import unicodedata

# NLP tools
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cleaning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


class TextCleaner:
    """Cleans and preprocesses text content for LLM fine-tuning."""
    
    def __init__(self, input_dir="raw", output_dir="cleaned"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Patterns for cleaning
        self.html_pattern = re.compile(r'<.*?>')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.special_chars_pattern = re.compile(r'[^\w\s.,?!;:()\[\]{}\-\'"–—]')
        self.multiple_spaces_pattern = re.compile(r'\s+')
        self.multiple_newlines_pattern = re.compile(r'\n{3,}')
        self.header_pattern = re.compile(r'^(#+|\*+)\s+')
        
    def normalize_unicode(self, text):
        """Normalize Unicode characters to ASCII where possible."""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    def clean_html(self, text):
        """Remove HTML tags."""
        return self.html_pattern.sub(' ', text)
    
    def clean_urls_and_emails(self, text):
        """Remove URLs and email addresses."""
        text = self.url_pattern.sub('[URL]', text)
        text = self.email_pattern.sub('[EMAIL]', text)
        return text
    
    def clean_special_chars(self, text):
        """Clean special characters but keep essential punctuation."""
        return self.special_chars_pattern.sub(' ', text)
    
    def clean_whitespace(self, text):
        """Clean excess whitespace."""
        text = self.multiple_spaces_pattern.sub(' ', text)
        text = self.multiple_newlines_pattern.sub('\n\n', text)
        return text.strip()
    
    def clean_markdown_headers(self, text):
        """Clean Markdown headers."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove header markers but keep the text
            match = self.header_pattern.search(line)
            if match:
                cleaned_lines.append(line[match.end():].strip())
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def fix_sentence_boundaries(self, text):
        """Fix sentence boundaries using NLTK's sentence tokenizer."""
        sentences = sent_tokenize(text)
        return ' '.join(sentences)
    
    def remove_non_prose(self, text):
        """
        Remove content that's not proper prose, like code blocks, tables,
        list items that are too short, etc.
        """
        lines = text.split('\n')
        filtered_lines = []
        
        code_block = False
        
        for line in lines:
            line = line.strip()
            
            # Skip code blocks
            if line.startswith('```') or line.startswith('~~~'):
                code_block = not code_block
                continue
            
            if code_block:
                continue
            
            # Skip short list items
            if re.match(r'^[\*\-\+\d+\.]\s+.{1,20}$', line):
                continue
                
            # Skip lines that look like table separators
            if re.match(r'^[\-\|=+]{3,}$', line):
                continue
                
            # Skip very short lines
            if len(line) < 15 and not line.endswith(('.', '?', '!')):
                continue
                
            filtered_lines.append(line)
            
        return '\n'.join(filtered_lines)
    
    def clean_text(self, text):
        """Apply all cleaning steps to the text."""
        if not text:
            return ""
            
        # Apply cleaning steps in sequence
        text = self.normalize_unicode(text)
        text = self.clean_html(text)
        text = self.clean_markdown_headers(text)
        text = self.clean_urls_and_emails(text)
        text = self.clean_special_chars(text)
        text = self.remove_non_prose(text)
        text = self.clean_whitespace(text)
        text = self.fix_sentence_boundaries(text)
        
        return text
    
    def clean_content_file(self, file_path):
        """Clean content from a JSON file."""
        try:
            logger.info(f"Cleaning content from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content_dict = json.load(f)
            
            # Clean the content field
            if "content" in content_dict:
                content_dict["original_content_length"] = len(content_dict["content"])
                content_dict["content"] = self.clean_text(content_dict["content"])
                content_dict["cleaned_content_length"] = len(content_dict["content"])
                
                # Record cleaning timestamp
                content_dict["cleaning_timestamp"] = datetime.now().isoformat()
                
                # Save to output directory
                output_path = self.output_dir / file_path.name
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(content_dict, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved cleaned content to {output_path}")
                return True
            
            logger.warning(f"No content field found in {file_path}")
            return False
        
        except Exception as e:
            logger.error(f"Error cleaning content from {file_path}: {e}")
            return False
    
    def clean_directory(self, max_workers=4):
        """Clean all JSON files in the input directory."""
        json_files = list(self.input_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.input_dir}")
            return 0, 0
        
        logger.info(f"Found {len(json_files)} JSON files to clean")
        
        successful = 0
        failed = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.clean_content_file, file_path): file_path for file_path in json_files}
            
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
        
        logger.info(f"Cleaning complete. Successful: {successful}, Failed: {failed}")
        return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess text for LLM fine-tuning")
    parser.add_argument("--input", "-i", default="raw", help="Input directory containing extracted content")
    parser.add_argument("--output", "-o", default="cleaned", help="Output directory for cleaned content")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    cleaner = TextCleaner(input_dir=args.input, output_dir=args.output)
    cleaner.clean_directory(max_workers=args.workers)


if __name__ == "__main__":
    main()
