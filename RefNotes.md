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

## Long Document Processing

'Tuner' to process very long chat history documents so that the data is cleaned and properly optimized/extracted/refined
etc.?

## codex

You can absolutely point Tuner at a folder full of raw chat‐transcript files and have it do the whole shebang—extract, clean,
    optimize, run the LLM‐driven “command extractor” stage, and spit out a ready‑to‑train dataset. Here’s a quick “how‐to”:

  1. Prep your chat logs
           • If they’re JSON already (e.g. `{sender, timestamp, text}`), write a tiny converter that emits Tuner’s “raw” JSON format:

      import json, uuid
               raw = {
                 "source": "chat_room_xyz",
                 "title": "April 2025 Stand‑up",
                 "date": "2025-04-15T10:00:00",
                 "content": "\n\n".join(f"{m['sender']}: {m['text']}" for m in messages)
               }
               with open("chat1.json","w") as f: json.dump(raw, f)

        and drop those `.json` files into a folder (say `my_chats_raw/`).
           • If it’s plain `.txt`, you can skip extraction and just put them in `my_chats_raw/` (Tuner’s extractor will read `.txt` as plain content).

  2. Run the full pipeline

        python fine_tuning_pipeline.py \
                 --dir my_chats_raw \
                 --base-dir finetuning_chats \
                 --instruction "Summarize this conversation thread:" \
                 --model gpt-4o \
                 --val-ratio 0.05

        Tuner will create:

        finetuning_chats/
                 raw/         ← your chat JSON/TXT
                 cleaned/     ← text_cleaner.py outputs
                 optimized/   ← content_optimizer.py outputs
                 refined/     ← command_extractor.py outputs
                 final/       ← dataset_creator.py outputs (openai/, anthropic/, …)
                 pipeline_metrics.json
  3. Tweak segmentation for chat style
           By default we chunk on blank‑line/paragraph boundaries, then sentences, but for short chat messages you may want smaller
            windows. You can pass:

        python content_optimizer.py \
                 --input finetuning_chats/cleaned \
                 --output finetuning_chats/optimized \
                 --min-segment 100 --max-segment 800 --overlap 20 \
                 --instruction "Summarize this chat segment:"

  4. Pull in only the bits you need
           If your logs are huge and you just want the most relevant snippets (e.g. “What decisions did we make about logging?”), you
    can embed each cleaned segment into a vector store and at run‑time retrieve top‑K before passing to the LLM. See the earlier “vector
     DB” recipe.

  5. Consume the final dataset
           Your `finetuning_chats/final/openai/train.jsonl` and `…/validation.jsonl` are now ready for `openai api fine_tunes.create …`
    (or any of the other formats).

  — That’s all there is to it. Drop your raw chat files into a folder, point Tuner at it, adjust segment lengths and instructions to suit, and you’ll get a cleaned, AI‑refined, ready‑to‑train dataset in minutes. user it's txt but the format is essentially what our chats look like - kind of mixed with code snippets and back and forth between us.

## Codex

  Since your logs are plain‑text transcripts with interleaved code blocks, you can treat them just like any other “raw” source in
    Tuner. Here’s a step‑by‑step:

  1. Drop your `.txt` files into a folder, e.g. `chat_logs_raw/`.
           Tuner’s extractor will pull in any `.txt` as-is (it doesn’t need JSON).
  2. Run the full pipeline:

      python fine_tuning_pipeline.py \
                 --dir chat_logs_raw \
                 --base-dir finetuning_chatlogs \
                 --instruction "Summarize this conversation segment:" \
                 --model gpt-3.5-turbo \
                 --val-ratio 0.05

        You’ll end up with:

        finetuning_chatlogs/
                 raw/       ← your .txt chat logs
                 cleaned/   ← cleaned transcripts (code fences preserved as text)
                 optimized/ ← segmented “chunks” (each chunk may contain a few messages + code)
                 refined/   ← AI‑extracted prompts/completions
                 final/     ← ready‑to‑train datasets (openai/, anthropic/, etc.)

  3. Tweak segmentation if you need smaller chunks around code:

        python content_optimizer.py \
                 --input finetuning_chatlogs/cleaned \
                 --output finetuning_chatlogs/optimized \
                 --min-segment 100 \
                 --max-segment 600 \
                 --overlap 20 \
                 --instruction "Summarize this chat snippet:"

        That ensures code snippets stay together and aren’t split mid‑block.

  4. Preserve code blocks in cleaning
           By default `text_cleaner.py` will strip non‑UTF8 junk but keeps back‑ticks. If you find code fences getting mangled, open

      `text_cleaner.py` and tweak the regex that removes Markdown-like formatting to ignore triple‑backtick fences.

  5. Extract/refine and build your dataset
           Once you have `optimized/`, run the command extractor and dataset creator exactly as shown in the README’s “Individual
    Component Usage” section:

  python command_extractor.py \
                 --input finetuning_chatlogs/optimized \
                 --output finetuning_chatlogs/refined \
                 --instruction "Summarize this chat snippet:" \
                 --model gpt-3.5-turbo

  python dataset_creator.py \
                 --input finetuning_chatlogs/refined \
                 --output finetuning_chatlogs/final \
                 --val-ratio 0.05 \
                 --instruction "Summarize this conversation segment:"

After that you’ll have final/openai/train.jsonl (and the other formats) ready for fine‑tuning.

1. Migrate to the new OpenAI Python API

        * Replaced      openai.ChatCompletion.create(…)

          with      openai.chat.completions.create(…, temperature=0)
        * Sets `temperature=0` for deterministic JSON output.

    2. Robust JSON‑only parsing

        * Strips markdown fences (` ``` ` / ```json`)
        * Finds the first “{” / last “}” to isolate the JSON blob
        * On parse errors, prints the raw (unfenced) text to stderr.

    3. Per‑file progress and summary prints

        * Prints “Found N JSON files in …”
        * Prints “Analyzing file i/N: <path>” for each chunk
        * At the end, prints how many actions / queries / explanations / decisions were collected.

    4. “Append” mode (via --append)

        * If the output file already exists, loads it, shows its counts, and adds new entries rather than
    overwriting.
        * Default behavior (no `--append`) still starts fresh.

    5. Timestamped directory output

        * If you pass a directory to `--output`, e.g. `--output chat-summaries/`, it will:
          • Create that directory if needed
          • Write a new file like `chat_summary_20250420_143212.json` inside it
          • Print its exact path for you
        * If you pass a file path instead, it behaves exactly as before.

    — — —

    Usage examples:

    A) Basic single run (overwrite or fresh file):

        python tools/chat_analyzer.py \
          --input finetuning_chatlogs/optimized \
          --output chat_summary.json \
          --model gpt-4o

    B) Append to an existing summary:

        python tools/chat_analyzer.py \
          --input new_chunks/optimized \
          --output chat_summary.json \
          --model gpt-4o \
          --append

    C) Keep every run in its own timestamped file:

        python tools/chat_analyzer.py \
          --input finetuning_chatlogs/optimized \
          --output chat-summaries/ \
          --model gpt-4o

    This drops files like chat-summaries/chat_summary_20250420_143212.json.
    