#!/usr/bin/env python3
"""
Content Extractor for LLM Fine-tuning Preparation
------------------------------------------------
This script extracts content from various sources (HTML, Markdown, PDF, etc.)
and converts it to a standardized format for further processing.
"""

import os
import re
import json
import argparse
from pathlib import Path
import concurrent.futures
from datetime import datetime
import logging

# For web content
import requests
from bs4 import BeautifulSoup
import trafilatura

# For file formats
import markdown
import pypandoc
import pdfplumber

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class ContentExtractor:
    """Extracts and standardizes content from various sources."""
    
    def __init__(self, output_dir="raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_from_url(self, url):
        """Extract content from a URL."""
        try:
            logger.info(f"Extracting from URL: {url}")
            
            # First try using trafilatura for better content extraction
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                content = trafilatura.extract(downloaded, include_comments=False, 
                                           include_tables=True, output_format="text")
                
                if content:
                    metadata = trafilatura.extract_metadata(downloaded)
                    return {
                        "source": url,
                        "title": metadata.title if metadata and metadata.title else "Unknown Title",
                        "date": metadata.date if metadata and metadata.date else datetime.now().isoformat(),
                        "content": content,
                        "source_type": "url"
                    }
            
            # Fallback to BeautifulSoup
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "Unknown Title"
            
            # Try to find the main content
            # This is a simple approach - you might need to adapt based on your docs' structure
            content_areas = soup.find_all(['article', 'main', 'div'], 
                                        class_=re.compile(r'content|article|post|entry'))
            
            if content_areas:
                main_content = max(content_areas, key=lambda x: len(x.get_text()))
                content = main_content.get_text(separator='\n', strip=True)
            else:
                # Fallback: get the body text
                content = soup.body.get_text(separator='\n', strip=True)
                
            return {
                "source": url,
                "title": title,
                "date": datetime.now().isoformat(),
                "content": content,
                "source_type": "url"
            }
        
        except Exception as e:
            logger.error(f"Error extracting from URL {url}: {e}")
            return None
    
    def extract_from_html(self, file_path):
        """Extract content from an HTML file."""
        try:
            logger.info(f"Extracting from HTML file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Try trafilatura first
            content = trafilatura.extract(html_content, include_comments=False, 
                                       include_tables=True, output_format="text")
            
            if content:
                metadata = trafilatura.extract_metadata(html_content)
                return {
                    "source": str(file_path),
                    "title": metadata.title if metadata and metadata.title else Path(file_path).stem,
                    "date": metadata.date if metadata and metadata.date else datetime.now().isoformat(),
                    "content": content,
                    "source_type": "html_file"
                }
            
            # Fallback to BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.title.string if soup.title else Path(file_path).stem
            
            # Try to find the main content
            content_areas = soup.find_all(['article', 'main', 'div'], 
                                        class_=re.compile(r'content|article|post|entry'))
            
            if content_areas:
                main_content = max(content_areas, key=lambda x: len(x.get_text()))
                content = main_content.get_text(separator='\n', strip=True)
            else:
                # Fallback: get the body text
                content = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
                
            return {
                "source": str(file_path),
                "title": title,
                "date": datetime.now().isoformat(),
                "content": content,
                "source_type": "html_file"
            }
        
        except Exception as e:
            logger.error(f"Error extracting from HTML file {file_path}: {e}")
            return None
    
    def extract_from_markdown(self, file_path):
        """Extract content from a Markdown file."""
        try:
            logger.info(f"Extracting from Markdown file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to HTML first for consistent processing
            html_content = markdown.markdown(md_content)
            
            # Extract using BeautifulSoup for simplicity
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try to extract title from first heading
            title = None
            first_heading = soup.find(['h1', 'h2'])
            if first_heading:
                title = first_heading.get_text(strip=True)
                # Remove the heading from the content to avoid duplication
                first_heading.extract()
            
            # If no heading, use filename
            if not title:
                title = Path(file_path).stem
                
            content = soup.get_text(separator='\n', strip=True)
            
            return {
                "source": str(file_path),
                "title": title,
                "date": datetime.now().isoformat(),
                "content": content,
                "source_type": "markdown_file"
            }
        
        except Exception as e:
            logger.error(f"Error extracting from Markdown file {file_path}: {e}")
            return None
    
    def extract_from_pdf(self, file_path):
        """Extract content from a PDF file."""
        try:
            logger.info(f"Extracting from PDF file: {file_path}")
            text_content = []
            
            with pdfplumber.open(file_path) as pdf:
                # Extract title from metadata if available
                title = None
                if pdf.metadata and 'Title' in pdf.metadata:
                    title = pdf.metadata['Title']
                
                if not title:
                    title = Path(file_path).stem
                
                # Extract text from each page
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            
            return {
                "source": str(file_path),
                "title": title,
                "date": datetime.now().isoformat(),
                "content": "\n\n".join(text_content),
                "source_type": "pdf_file"
            }
        
        except Exception as e:
            logger.error(f"Error extracting from PDF file {file_path}: {e}")
            return None
    
    def extract_from_docx(self, file_path):
        """Extract content from a DOCX file."""
        try:
            logger.info(f"Extracting from DOCX file: {file_path}")
            
            # Use pandoc to convert DOCX to plain text
            text = pypandoc.convert_file(str(file_path), 'plain', format='docx')
            
            # Use the filename as the title
            title = Path(file_path).stem
            
            return {
                "source": str(file_path),
                "title": title,
                "date": datetime.now().isoformat(),
                "content": text,
                "source_type": "docx_file"
            }
        
        except Exception as e:
            logger.error(f"Error extracting from DOCX file {file_path}: {e}")
            return None
            
    def extract_from_text(self, file_path):
        """Extract content from a plain text file."""
        try:
            logger.info(f"Extracting from text file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text_content = f.read()
            
            # Use the filename as the title
            title = Path(file_path).stem
            
            # Try to extract title from first line if it looks like a title
            lines = text_content.splitlines()
            if lines and len(lines[0].strip()) < 80:  # Reasonable title length
                title = lines[0].strip()
                # Remove the title line from content to avoid duplication
                text_content = '\n'.join(lines[1:])
            
            return {
                "source": str(file_path),
                "title": title,
                "date": datetime.now().isoformat(),
                "content": text_content,
                "source_type": "text_file"
            }
        
        except Exception as e:
            logger.error(f"Error extracting from text file {file_path}: {e}")
            return None
    
    def save_content(self, content_dict):
        """Save extracted content to a JSON file."""
        if not content_dict:
            return False
        
        try:
            # Create a valid filename from the title
            safe_title = re.sub(r'[^\w\s-]', '', content_dict["title"]).strip().lower()
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            
            # Add timestamp to ensure uniqueness
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{safe_title}_{timestamp}.json"
            
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(content_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved content to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            return False
    
    def process_source(self, source):
        """Process a source based on its type and save the extracted content."""
        content_dict = None
        
        if source.startswith(('http://', 'https://')):
            content_dict = self.extract_from_url(source)
        else:
            source_path = Path(source)
            if not source_path.exists():
                logger.warning(f"Source does not exist: {source}")
                return False
            
            suffix = source_path.suffix.lower()
            
            if suffix in ['.html', '.htm']:
                content_dict = self.extract_from_html(source_path)
            elif suffix in ['.md', '.markdown']:
                content_dict = self.extract_from_markdown(source_path)
            elif suffix == '.pdf':
                content_dict = self.extract_from_pdf(source_path)
            elif suffix in ['.docx', '.doc']:
                content_dict = self.extract_from_docx(source_path)
            elif suffix in ['.txt', '.text']:
                content_dict = self.extract_from_text(source_path)
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return False
        
        return self.save_content(content_dict)
    
    def process_sources_parallel(self, sources, max_workers=4):
        """Process multiple sources in parallel."""
        successful = 0
        failed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_source, source): source for source in sources}
            
            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error processing {source}: {e}")
                    failed += 1
        
        logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")
        return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Extract content from various sources for LLM fine-tuning")
    parser.add_argument("--output", "-o", default="raw", help="Output directory for extracted content")
    
    # Source input options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--sources", "-s", nargs="+", help="List of sources (URLs or file paths)")
    source_group.add_argument("--file", "-f", help="File containing sources (one per line)")
    source_group.add_argument("--dir", "-d", help="Directory to recursively scan for compatible files")
    
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    extractor = ContentExtractor(output_dir=args.output)
    
    # Collect sources
    sources = []
    
    if args.sources:
        sources = args.sources
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f if line.strip()]
    elif args.dir:
        dir_path = Path(args.dir)
        extensions = ['.html', '.htm', '.md', '.markdown', '.pdf', '.docx', '.doc', '.txt', '.text']
        
        for ext in extensions:
            sources.extend([str(p) for p in dir_path.rglob(f"*{ext}")])
    
    # Process the sources
    if sources:
        logger.info(f"Found {len(sources)} sources to process")
        extractor.process_sources_parallel(sources, max_workers=args.workers)
    else:
        logger.warning("No sources found")


if __name__ == "__main__":
    main()
