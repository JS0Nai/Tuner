#!/usr/bin/env python3
"""
Chat Analyzer
-------------
Takes optimized JSON chunks from Tuner (content_optimizer output) and classifies
each chunk into four categories: actions, queries, explanations, decisions.
Aggregates them into a summary JSON.

Usage:
    python tools/chat_analyzer.py --input optimized_dir --output chat_summary.json

Prerequisites:
    - Python 3.8+
    - openai Python package
    - OPENAI_API_KEY environment variable set
"""

import json
import glob
import argparse
import openai

DEFAULT_MODEL = "gpt-4o"

ANALYSIS_PROMPT = """
I’m going to feed you chunks of a user↔assistant chat.
For each chunk, return a single JSON object with exactly four arrays:
  1. "actions": things someone did (e.g., "ran pip install ...")
  2. "queries": user questions or requests
  3. "explanations": places the assistant explained something
  4. "decisions": agreed next steps or environment choices

Make the lists chronological (in chunk order) and output only valid JSON.
"""

def analyze_chunk(chunk_text, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user", "content": chunk_text}
        ]
    )
    return json.loads(response.choices[0].message.content)

def main(input_dir, output_file, model):
    summary = {
        "actions": [],
        "queries": [],
        "explanations": [],
        "decisions": []
    }

    pattern = f"{input_dir.rstrip('/')}/*.json"
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Try common fields, fall back to entire JSON
        chunk = data.get("segment") or data.get("content") or json.dumps(data)
        result = analyze_chunk(chunk, model)
        for key in summary:
            summary[key].extend(result.get(key, []))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze optimized chat chunks into categorized lists."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Directory containing optimized JSON chunks."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSON summary file."
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help="OpenAI model to use for analysis."
    )
    args = parser.parse_args()
    main(args.input, args.output, args.model)