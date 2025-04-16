#!/usr/bin/env python3
"""
Dataset Creator for LLM Fine-tuning
----------------------------------
This script creates the final dataset for language model fine-tuning
in various formats compatible with different training frameworks.
"""

import os
import json
import argparse
from pathlib import Path
import logging
import random
from datetime import datetime
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_creation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


class DatasetCreator:
    """Creates fine-tuning datasets in various formats."""
    
    def __init__(self, input_dir="optimized", output_dir="final", instruction="Continue writing in the style of the author:"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.instruction = instruction
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different formats
        self.openai_dir = self.output_dir / "openai"
        self.openai_dir.mkdir(exist_ok=True)
        
        self.anthropic_dir = self.output_dir / "anthropic"
        self.anthropic_dir.mkdir(exist_ok=True)
        
        self.huggingface_dir = self.output_dir / "huggingface"
        self.huggingface_dir.mkdir(exist_ok=True)
        
        self.llama_dir = self.output_dir / "llama"
        self.llama_dir.mkdir(exist_ok=True)
        
        self.jsonl_dir = self.output_dir / "jsonl"
        self.jsonl_dir.mkdir(exist_ok=True)
        
        # Metadata collection
        self.metadata = {
            "dataset_creation_time": datetime.now().isoformat(),
            "total_examples": 0,
            "total_segments": 0,
            "total_tokens_estimate": 0,
            "source_files": [],
            "split": {
                "train": 0,
                "validation": 0
            },
            "style_markers_summary": {},
        }
    
    def collect_training_examples(self):
        """Collect all training examples from optimized files."""
        json_files = list(self.input_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.input_dir}")
            return []
        
        logger.info(f"Found {len(json_files)} optimized files")
        
        all_examples = []
        style_markers_collection = {}
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    optimized_data = json.load(f)
                
                # Collect basic info
                examples_count = len(optimized_data.get("training_examples", []))
                source_info = {
                    "filename": file_path.name,
                    "title": optimized_data.get("title", ""),
                    "original_source": optimized_data.get("original_source", ""),
                    "examples_count": examples_count
                }
                self.metadata["source_files"].append(source_info)
                
                # Collect style markers
                if "style_markers" in optimized_data:
                    for marker_type, markers in optimized_data["style_markers"].items():
                        if marker_type not in style_markers_collection:
                            style_markers_collection[marker_type] = []
                        style_markers_collection[marker_type].append(markers)
                
                # Add examples
                all_examples.extend(optimized_data.get("training_examples", []))
                
                # Update counts
                self.metadata["total_examples"] += 1
                self.metadata["total_segments"] += examples_count
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Process collected style markers into summary
        self.process_style_markers_summary(style_markers_collection)
        
        # Estimate token count (rough approximation)
        total_text = ""
        for example in all_examples:
            if "segment" in example:
                total_text += example["segment"]
        
        # Estimate: ~1 token per 4 characters
        self.metadata["total_tokens_estimate"] = len(total_text) // 4
        
        logger.info(f"Collected {len(all_examples)} training examples")
        return all_examples
    
    def process_style_markers_summary(self, style_markers_collection):
        """Process collected style markers into a summary."""
        summary = {}
        
        # Process numeric markers (averages)
        numeric_markers = ["avg_sentence_length", "avg_paragraph_length"]
        for marker in numeric_markers:
            if marker in style_markers_collection:
                values = style_markers_collection[marker]
                # Filter out any non-numeric values
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    summary[marker] = sum(numeric_values) / len(numeric_values)
        
        # Process list markers (most common)
        list_markers = ["transition_phrases", "common_phrases"]
        for marker in list_markers:
            if marker in style_markers_collection:
                # Flatten the list of lists
                all_items = []
                for item_list in style_markers_collection[marker]:
                    all_items.extend(item_list)
                
                # Count frequencies
                item_counter = {}
                for item_tuple in all_items:
                    if isinstance(item_tuple, (list, tuple)) and len(item_tuple) >= 2:
                        item, count = item_tuple[0], item_tuple[1]
                        item_counter[item] = item_counter.get(item, 0) + count
                
                # Get top 15
                summary[marker] = sorted(
                    item_counter.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:15]
        
        # Process stylistic elements
        if "stylistic_elements" in style_markers_collection:
            element_counter = {}
            per_1000_totals = {}
            counts = {}
            
            for elements_list in style_markers_collection["stylistic_elements"]:
                for element_dict in elements_list:
                    element = element_dict.get("element")
                    count = element_dict.get("count", 0)
                    per_1000 = element_dict.get("per_1000_chars", 0)
                    
                    if element:
                        element_counter[element] = element_counter.get(element, 0) + 1
                        per_1000_totals[element] = per_1000_totals.get(element, 0) + per_1000
                        counts[element] = counts.get(element, 0) + count
            
            # Calculate averages
            summary["stylistic_elements"] = []
            for element, occurrences in element_counter.items():
                avg_per_1000 = per_1000_totals[element] / occurrences
                summary["stylistic_elements"].append({
                    "element": element,
                    "avg_per_1000_chars": avg_per_1000,
                    "total_count": counts[element]
                })
            
            # Sort by frequency
            summary["stylistic_elements"].sort(key=lambda x: x["total_count"], reverse=True)
        
        self.metadata["style_markers_summary"] = summary
    
    def split_train_validation(self, examples, val_ratio=0.1):
        """Split examples into training and validation sets."""
        random.shuffle(examples)
        val_count = max(1, int(len(examples) * val_ratio))
        
        train_examples = examples[val_count:]
        val_examples = examples[:val_count]
        
        self.metadata["split"]["train"] = len(train_examples)
        self.metadata["split"]["validation"] = len(val_examples)
        
        logger.info(f"Split: {len(train_examples)} training, {len(val_examples)} validation examples")
        
        return train_examples, val_examples
    
    def create_openai_format(self, train_examples, val_examples):
        """Create dataset in OpenAI fine-tuning format."""
        logger.info("Creating OpenAI format dataset")
        
        # Function to convert to OpenAI format
        def format_example(example):
            # OpenAI uses a simple JSONL format with prompt and completion fields
            return {
                "prompt": example["prompt"],
                "completion": example["completion"]
            }
        
        # Create train file
        train_path = self.openai_dir / "train.jsonl"
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(format_example(example)) + '\n')
        
        # Create validation file
        val_path = self.openai_dir / "validation.jsonl"
        with open(val_path, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(format_example(example)) + '\n')
        
        # Create info file
        info_path = self.openai_dir / "info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                "format": "OpenAI fine-tuning",
                "train_examples": len(train_examples),
                "validation_examples": len(val_examples),
                "created": datetime.now().isoformat(),
                "notes": "For use with OpenAI's fine-tuning API"
            }, f, indent=2)
        
        logger.info(f"OpenAI format dataset created at {self.openai_dir}")
    
    def create_anthropic_format(self, train_examples, val_examples):
        """Create dataset in Anthropic Claude fine-tuning format."""
        logger.info("Creating Anthropic format dataset")
        
        # Function to convert to Anthropic format (similar to ChatML)
        def format_example(example):
            # Get the chat format if available, or create one
            if "chat_format" in example:
                messages = example["chat_format"]
            else:
                # Use dynamic instruction for system and user messages
                inst_clean = self.instruction.rstrip(":").strip()
                sys_msg = f"You are an assistant that {inst_clean.lower()}"
                messages = [
                    {"role": "system",    "content": sys_msg},
                    {"role": "user",      "content": self.instruction},
                    {"role": "assistant", "content": example["completion"]}
                ]
            
            # Format according to Anthropic's expected structure
            return {
                "messages": messages
            }
        
        # Create train file
        train_path = self.anthropic_dir / "train.jsonl"
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(format_example(example)) + '\n')
        
        # Create validation file
        val_path = self.anthropic_dir / "validation.jsonl"
        with open(val_path, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(format_example(example)) + '\n')
        
        # Create info file
        info_path = self.anthropic_dir / "info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                "format": "Anthropic Claude fine-tuning",
                "train_examples": len(train_examples),
                "validation_examples": len(val_examples),
                "created": datetime.now().isoformat(),
                "notes": "For use with Anthropic's Claude fine-tuning API"
            }, f, indent=2)
        
        logger.info(f"Anthropic format dataset created at {self.anthropic_dir}")
    
    def create_huggingface_format(self, train_examples, val_examples):
        """Create dataset in Hugging Face format."""
        logger.info("Creating Hugging Face format dataset")
        
        # For HuggingFace, we'll create a dataset with appropriate fields
        train_data = {
            "text": [],  # For text-only models
            "input_ids": [],  # Will be computed by tokenizer
            "prompt": [],
            "completion": [],
            "metadata": []
        }
        
        val_data = {
            "text": [],
            "input_ids": [],
            "prompt": [],
            "completion": [],
            "metadata": []
        }
        
        for example in train_examples:
            train_data["text"].append(example["segment"])
            train_data["prompt"].append(example["prompt"])
            train_data["completion"].append(example["completion"])
            train_data["metadata"].append(example.get("metadata", {}))
        
        for example in val_examples:
            val_data["text"].append(example["segment"])
            val_data["prompt"].append(example["prompt"])
            val_data["completion"].append(example["completion"])
            val_data["metadata"].append(example.get("metadata", {}))
        
        # Save as JSON files that can be loaded with datasets library
        train_path = self.huggingface_dir / "train.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f)
        
        val_path = self.huggingface_dir / "validation.json"
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f)
        
        # Create a README.md file with instructions
        readme_path = self.huggingface_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("""# Hugging Face Dataset for Style Fine-tuning

This dataset is prepared for fine-tuning language models using the Hugging Face ecosystem.

## Loading the Dataset

```python
from datasets import load_dataset

# Load from local files
dataset = load_dataset('json', data_files={
    'train': 'train.json',
    'validation': 'validation.json'
})

# Alternatively, if you've pushed this to the Hugging Face Hub:
# dataset = load_dataset('your-username/dataset-name')
```

## Format Details

The dataset contains the following fields:

- `text`: The full text segment
- `prompt`: The prompt text
- `completion`: The completion text
- `metadata`: Additional metadata about the sample

## Fine-tuning Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('your-base-model')
model = AutoModelForCausalLM.from_pretrained('your-base-model')

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation']
)

# Train model
trainer.train()
```
""")
        
        logger.info(f"Hugging Face format dataset created at {self.huggingface_dir}")
    
    def create_llama_format(self, train_examples, val_examples):
        """Create dataset in Llama fine-tuning format."""
        logger.info("Creating LLaMa format dataset")
        
        # Function to convert to Llama format
        def format_example(example):
            # LLaMa-style format (similar to Alpaca dataset)
            inst_text = self.instruction.rstrip(":").strip()
            return {
                "instruction": inst_text,
                "input": example["prompt"],
                "output": example["completion"]
            }
        
        # Create train file
        train_path = self.llama_dir / "train.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump([format_example(ex) for ex in train_examples], f, indent=2)
        
        # Create validation file
        val_path = self.llama_dir / "val.json"
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump([format_example(ex) for ex in val_examples], f, indent=2)
        
        # Create info file
        info_path = self.llama_dir / "info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                "format": "LLaMa/Alpaca-style fine-tuning",
                "train_examples": len(train_examples),
                "validation_examples": len(val_examples),
                "created": datetime.now().isoformat(),
                "notes": "For use with LLaMa/Alpaca-compatible fine-tuning frameworks"
            }, f, indent=2)
        
        logger.info(f"LLaMa format dataset created at {self.llama_dir}")
    
    def create_jsonl_format(self, train_examples, val_examples):
        """Create dataset in raw JSONL format."""
        logger.info("Creating raw JSONL format dataset")
        
        # Create train file with full examples
        train_path = self.jsonl_dir / "train.jsonl"
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')
        
        # Create validation file with full examples
        val_path = self.jsonl_dir / "validation.jsonl"
        with open(val_path, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Raw JSONL format dataset created at {self.jsonl_dir}")
    
    def save_metadata(self):
        """Save overall dataset metadata."""
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Dataset metadata saved to {metadata_path}")
    
    def create_all_formats(self, val_ratio=0.1):
        """Create datasets in all formats."""
        # Collect all examples
        all_examples = self.collect_training_examples()
        
        if not all_examples:
            logger.error("No examples found. Dataset creation failed.")
            return False
        
        # Split into train and validation sets
        train_examples, val_examples = self.split_train_validation(all_examples, val_ratio)
        
        # Create each format
        self.create_openai_format(train_examples, val_examples)
        self.create_anthropic_format(train_examples, val_examples)
        self.create_huggingface_format(train_examples, val_examples)
        self.create_llama_format(train_examples, val_examples)
        self.create_jsonl_format(train_examples, val_examples)
        
        # Save overall metadata
        self.save_metadata()
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Create final datasets for LLM fine-tuning")
    parser.add_argument("--input", "-i", default="optimized", help="Input directory containing optimized content")
    parser.add_argument("--output", "-o", default="final", help="Output directory for final datasets")
    parser.add_argument("--val-ratio", "-v", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--instruction", "-t", default="Continue writing in the style of the author:", help="Instruction/prompt for each example")
    
    args = parser.parse_args()
    
    creator = DatasetCreator(input_dir=args.input, output_dir=args.output, instruction=args.instruction)
    creator.create_all_formats(val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
