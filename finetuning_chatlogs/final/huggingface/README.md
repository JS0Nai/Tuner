# Hugging Face Dataset for Style Fine-tuning

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
