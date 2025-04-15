# Tuner Project Notes

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
python fine_tuning_pipeline.py --sources https://example.com/post1 https://example.com/post2

# Process files from a directory
python fine_tuning_pipeline.py --dir /path/to/content

# Process a single file
python fine_tuning_pipeline.py --sources "/path/to/file.md"
```

### Running Individual Components

```bash
# Content Extraction
python content_extractor.py --output finetuning/raw --sources "/path/to/file.md"

# Text Cleaning
python text_cleaner.py --input finetuning/raw --output finetuning/cleaned

# Content Optimization
python content_optimizer.py --input finetuning/cleaned --output finetuning/optimized

# Dataset Creation
python dataset_creator.py --input finetuning/optimized --output finetuning/final --val-ratio 0.1
```

## Project Structure

- `content_extractor.py`: Extracts text from various sources
- `text_cleaner.py`: Cleans and preprocesses text
- `content_optimizer.py`: Optimizes content for fine-tuning
- `dataset_creator.py`: Creates datasets in different formats
- `fine_tuning_pipeline.py`: Orchestrates the full pipeline
- `web_interface.py`: Provides a web UI

## Output Structure

```bash
finetuning/
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
  ./tuner/bin/pip install lxml_html_clean
  ```

  How to ingest this into a fine-tuning model

  Different LLM providers have specific ways to use these datasets:

1. OpenAI Fine-tuning:
    - Use the JSONL files in the OpenAI format (found in /finetuning/final/openai/)
    - Upload via their API: openai api fine_tunes.create --training_file=train.jsonl --validation_file=validation.jsonl
  --model=gpt-3.5-turbo-0613
    - Or through their web interface in the fine-tuning section
2. Anthropic Claude Fine-tuning:
    - Use the files in /finetuning/final/anthropic/
    - Submit through Anthropic's fine-tuning API or portal (they have a specific process for enterprise customers)
3. Hugging Face Models:
    - Use files in /finetuning/final/huggingface/
    - Load with their dataset library: dataset = load_dataset('json', data_files={'train': 'train.json', 'validation': 'validation.json'})
    - Use with a trainer: trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset['train'])
4. LLaMA/Open Source Models:
    - Use the LLaMA format files in /finetuning/final/llama/
    - With libraries like LLaMA-Factory or other open-source training frameworks

  Using the application:

1. Any text-based content:
    - FAQs (as you've just demonstrated)
    - Technical documentation
    - Product descriptions
    - Scripts and dialogues
    - Academic papers
    - Legal documents
    - Marketing copy
2. Various file formats:
    - Markdown (.md)
    - Text (.txt)
    - HTML (.html)
    - PDF (.pdf)
    - Word documents (.docx)

  The system is designed to extract, clean, and segment any text-based content.

  How the code could be modified (hypothetically)

  If you wanted to specialize the system for different content types, these would be the key areas to modify:

  1. Content Extractor:
    - Add support for specialized formats (e.g., CSV, XML, JSON)
    - Optimize extraction for specific document types with custom parsers
  2. Text Cleaner:
    - Add domain-specific cleaning rules (e.g., preserving equations for scientific papers)
    - Create content-specific preprocessing (e.g., legal document formatting)
  3. Content Optimizer:
    - Adjust segmentation parameters based on content type (technical docs might need longer segments)
    - Add specialized prompt templates for different use cases
  4. Dataset Creator:
    - Create new output formats for specific training frameworks
    - Add specialized metadata fields relevant to your content type

  The beauty of the system is its modular design - you could extend it for specific use cases without breaking the core functionality.

  Practical use cases

  Some interesting applications you could use this for right now:

  1. Customer support materials: Convert support docs into fine-tuning data for customer service bots
  2. Internal documentation: Train models on your company's processes and procedures
  3. Product information: Create product-specific assistants that know your catalog
  4. Technical writing: Capture the style of technical documentation
  5. Legal documents: Format contract templates for training legal assistants

  The system handles the heavy lifting of converting raw text into properly formatted training data, which is one of the more tedious parts
  of fine-tuning.
