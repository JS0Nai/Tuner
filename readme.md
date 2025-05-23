# Content Preparation System for LLM Fine-Tuning

![Banner](media/banner.png)

This system provides a comprehensive pipeline for preparing, cleaning, and optimizing content for fine-tuning language models on your writing style and tone.

## Overview

The system takes your existing content from various sources (URLs, files, directories) and processes it through several stages to create high-quality datasets suitable for fine-tuning different language models.

### Pipeline Stages

1. **Content Extraction**: Extracts text content from various sources (HTML, Markdown, PDF, DOCX).
2. **Text Cleaning**: Cleans and preprocesses the extracted content to improve quality.
3. **Content Optimization**: Segments and optimizes the cleaned content for fine-tuning.
4. **AI-driven Command Extraction/Refinement**: Uses an AI model (e.g., OpenAI API) to extract and refine prompt–completion structures from optimized content.
5. **Dataset Creation**: Creates fine-tuning datasets in various formats compatible with different training frameworks.

### Supported Input Formats

- Webpages (URLs)
- HTML files
- Markdown files
- PDF documents
- Microsoft Word documents (DOCX)
- Plain text files (TXT)

### Supported Output Formats

- OpenAI fine-tuning format
- Anthropic Claude fine-tuning format
- Hugging Face dataset format
- LLaMA/Alpaca-style format
- Raw JSONL format

## Installation

### Prerequisites

        - Python 3.8 or higher
        - pip (Python package installer)

### API Key Configuration

The OpenAI API key must be available in the `OPENAI_API_KEY` environment variable. You have several options:

1. Add it to your shell startup (e.g., in `~/.bashrc` or `~/.zshrc`):
   ```bash
   export OPENAI_API_KEY="sk-...your_key_here..."
   ```

2. Use a `.env` file and `python-dotenv`:
   - Create a file named `.env` in this project root:
     ```text
     OPENAI_API_KEY=sk-...your_key_here...
     ```
   - Install python-dotenv:
     ```bash
     pip install python-dotenv
     ```
   - At the top of any Python script that uses OpenAI, add:
     ```python
     from dotenv import load_dotenv
     load_dotenv()
     ```

3. Use direnv for per-directory environments (recommended for scoped keys):
   - Ensure direnv is installed (https://direnv.net/).
   - We provide a sample `.envrc` file in the project root; edit it to include your key.
   - In the project root, run:
     ```bash
     direnv allow
     ```
   Now, whenever you `cd` into this directory, `OPENAI_API_KEY` will be set automatically.

### Setup

### Setup

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Continued ...
3. Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

You can use this system in two ways:

### 1. Web Interface (Recommended)

The web interface provides an easy-to-use UI for managing the fine-tuning preparation process.

```bash
python web_interface.py
```

This will start a local web server and open the interface in your default web browser.
From there, you can:

- Select your content source (URLs, file, or directory)
- Configure pipeline parameters
- Monitor pipeline progress
- Download the resulting datasets

### 2. Command Line Interface

You can also run the pipeline directly from the command line:

```bash
python fine_tuning_pipeline.py [OPTIONS]
```

Examples:
  python fine_tuning_pipeline.py --sources file1.txt file2.md
  python fine_tuning_pipeline.py --file list_of_sources.txt
  python fine_tuning_pipeline.py --dir /path/to/content_directory

#### Required Arguments

One of the following source options is required:

- `--sources`, `-s`: List of sources (URLs or file paths)
- `--file`, `-f`: File containing sources (one per line)
- `--dir`, `-d`: Directory to recursively scan for compatible files

#### Optional Arguments

- `--base-dir`: Base directory for the pipeline (default: "finetuning")
- `--val-ratio`, `-v`: Validation set ratio (default: 0.1)
- `--instruction`, `-t`: Instruction or prompt for command extraction/refinement (default: "Continue writing in the style of the author:")
- `--model`, `-m`: OpenAI model to use for AI-driven command extraction (default: "gpt-3.5-turbo")

#### Examples

Process a list of URLs:

```bash
python fine_tuning_pipeline.py --sources https://example.com/post1 https://example.com/post2
```

Process URLs from a file:

```bash
python fine_tuning_pipeline.py --file urls.txt
```

Process files from a directory:

```bash
python fine_tuning_pipeline.py --dir /path/to/content
```

## Individual Component Usage

You can also run each stage of the pipeline separately:

### 1. Content Extraction

```bash
python content_extractor.py --output raw [SOURCE OPTIONS]
```

### 2. Text Cleaning

```bash
python text_cleaner.py --input raw --output cleaned
```

### 3. Content Optimization

```bash
python content_optimizer.py --input cleaned --output optimized
```

### 4. AI-driven Command Extraction/Refinement

```bash
python command_extractor.py --input optimized --output refined \
  --instruction "Continue writing in the style of the author:" \
  --model gpt-3.5-turbo
```

### 5. Dataset Creation

```bash
python dataset_creator.py --input refined --output final \
  --val-ratio 0.1 --instruction "Continue writing in the style of the author:"
```

### 6. Chat Analysis (Optional)

After generating optimized segments (e.g., from chat logs), you can classify them into actions, queries, explanations, and decisions:

```bash
python tools/chat_analyzer.py \
  --input path/to/optimized \
  --output chat_summary.json \
  --model gpt-4o
```

Inspect `chat_summary.json` for a structure like:
```json
{
  "actions": [...],
  "queries": [...],
  "explanations": [...],
  "decisions": [...]
}
```

To customize categories or prompt phrasing, edit the `ANALYSIS_PROMPT` in `tools/chat_analyzer.py`.

## Fine-Tuning with the Generated Datasets

After running the pipeline, you'll have datasets in various formats ready for fine-tuning:

### OpenAI Fine-Tuning

```bash
# Using OpenAI CLI
openai api fine_tunes.create \
  --training_file=final/openai/train.jsonl \
  --validation_file=final/openai/validation.jsonl \
  --model=gpt-3.5-turbo-0613
```

### Anthropic Claude Fine-Tuning

Use the `final/anthropic/train.jsonl` and `final/anthropic/validation.jsonl` files with Anthropic's fine-tuning API.

### Hugging Face Fine-Tuning

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load dataset
dataset = load_dataset('json', data_files={
    'train': 'final/huggingface/train.json',
    'validation': 'final/huggingface/validation.json'
})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('your-base-model')
model = AutoModelForCausalLM.from_pretrained('your-base-model')

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Configure training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation']
)

# Train model
trainer.train()
```

### LLaMA/Alpaca Fine-Tuning

Use the `final/llama/train.json` and `final/llama/val.json` files with LLaMA fine-tuning scripts.

## Directory Structure

```txt
finetuning/
├── raw/            # Raw extracted content
├── cleaned/        # Cleaned content
├── optimized/      # Optimized content for fine-tuning
├── refined/        # AI-driven command extraction/refinement output
└── final/          # Final datasets in various formats
    ├── openai/     # OpenAI format
    ├── anthropic/  # Anthropic Claude format
    ├── huggingface/ # Hugging Face format
    ├── llama/      # LLaMA/Alpaca format
    └── jsonl/      # Raw JSONL format
``` 

## Advanced Configuration

### Content Extraction

- `--workers`: Number of parallel workers (default: 4)

### Text Cleaning

- `--workers`: Number of parallel workers (default: 4)

### Content Optimization

- `--min-segment`: Minimum segment length (default: 200)
- `--max-segment`: Maximum segment length (default: 2000)
- `--overlap`: Segment overlap size (default: 50)
- `--workers`: Number of parallel workers (default: 4)

### Dataset Creation

- `--val-ratio`: Validation set ratio (default: 0.1)

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Make sure all dependencies are installed using `pip install -r requirements.txt`
2. **spaCy model error**: Install the spaCy model with `python -m spacy download en_core_web_sm`
3. **Permission errors**: Ensure you have write permissions to the output directories
4. **Memory errors**: For large datasets, reduce the number of parallel workers

### Logs

Log files are created for each component:

- `extraction.log` - Content extraction logs
- `cleaning.log` - Text cleaning logs
- `optimization.log` - Content optimization logs
- `dataset_creation.log` - Dataset creation logs
- `pipeline.log` - Complete pipeline logs
- `web_interface.log` - Web interface logs

## Contributing

Contributions to improve this system are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<img width="961" alt="image" src="https://github.com/user-attachments/assets/8179c88a-1b56-43e4-b8b9-afdfb1678b39" />

