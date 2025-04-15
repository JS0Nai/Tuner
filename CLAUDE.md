# Blog Tuner Project Notes

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download required spaCy model
python -m spacy download en_core_web_sm

# Install additional dependency if needed
pip install lxml_html_clean
```

### Running the Web Interface
```bash
python web_interface.py
# This will start a server at http://127.0.0.1:5000
```

### Running the Pipeline via Command Line
```bash
# Process URLs
python fine_tuning_pipeline.py --sources https://example.com/blog/post1 https://example.com/blog/post2

# Process files from a directory
python fine_tuning_pipeline.py --dir /path/to/blog/content

# Process a single file
python fine_tuning_pipeline.py --sources "/path/to/file.md"
```

### Running Individual Components
```bash
# Content Extraction
python content_extractor.py --output blog_finetuning/raw --sources "/path/to/file.md"

# Text Cleaning
python text_cleaner.py --input blog_finetuning/raw --output blog_finetuning/cleaned

# Content Optimization
python content_optimizer.py --input blog_finetuning/cleaned --output blog_finetuning/optimized

# Dataset Creation
python dataset_creator.py --input blog_finetuning/optimized --output blog_finetuning/final --val-ratio 0.1
```

## Project Structure
- `content_extractor.py`: Extracts text from various sources
- `text_cleaner.py`: Cleans and preprocesses text
- `content_optimizer.py`: Optimizes content for fine-tuning
- `dataset_creator.py`: Creates datasets in different formats
- `fine_tuning_pipeline.py`: Orchestrates the full pipeline
- `web_interface.py`: Provides a web UI

## Output Structure
```
blog_finetuning/
├── raw/            # Raw extracted content
├── cleaned/        # Cleaned content
├── optimized/      # Optimized content for fine-tuning
├── pipeline_metrics.json    # Performance metrics
└── final/          # Final datasets in various formats
    ├── openai/     # OpenAI format
    ├── anthropic/  # Anthropic Claude format
    ├── huggingface/ # Hugging Face format
    ├── llama/      # LLaMA/Alpaca format
    ├── jsonl/      # Raw JSONL format
    └── dataset_metadata.json # Dataset statistics
```

## Output Formats
- OpenAI fine-tuning format
- Anthropic Claude fine-tuning format
- Hugging Face dataset format
- LLaMA/Alpaca-style format
- Raw JSONL format

## Known Issues and Solutions
- If you encounter `ImportError: lxml.html.clean module is now a separate project lxml_html_clean`, install the missing dependency:
  ```bash
  pip install lxml_html_clean
  ```
- If running in a virtual environment, make sure to install the package in the environment:
  ```bash
  ./blogtuner/bin/pip install lxml_html_clean
  ```