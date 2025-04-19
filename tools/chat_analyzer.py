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
    - python-dotenv (in requirements.txt)
    - a `.env` file or $OPENAI_API_KEY set in env
"""

from dotenv import load_dotenv
load_dotenv()
import json
import glob
import argparse
import os
import sys
import openai
from datetime import datetime

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
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user", "content": chunk_text}
        ],
        temperature=0
    )
    # Extract and clean up the JSON from the model response
    raw = response.choices[0].message.content
    # Strip markdown code fences if present
    if raw.strip().startswith("```"):
        # remove leading/trailing fence lines
        lines = raw.splitlines()
        # drop any lines that are solely backticks or ```json
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw = "\n".join(lines)
    # Extract the first JSON object in the text
    start = raw.find("{")
    end = raw.rfind("}")
    content = raw[start:end+1] if start != -1 and end != -1 else raw
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Error parsing JSON from model response:", file=sys.stderr)
        print(raw, file=sys.stderr)
        raise

def main(input_dir, output_file, model):
    # Determine final output path: directory handling
    orig_output = output_file
    if os.path.isdir(orig_output) or orig_output.endswith(os.sep):
        out_dir = orig_output if os.path.isdir(orig_output) else orig_output.rstrip(os.sep)
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(out_dir, f"chat_summary_{timestamp}.json")
        print(f"Output directory detected. Writing to {output_file}")
    else:
        parent = os.path.dirname(orig_output)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        output_file = orig_output

    # Load existing summary if appending (only applies to file outputs)
    summary = None
    if getattr(main, 'append', False) and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            print(f"Appending to existing summary ({output_file}): ", end='')
            print({k: len(v) for k, v in summary.items()})
        except Exception:
            print(f"Warning: failed to load existing summary at {output_file}, starting fresh", file=sys.stderr)
            summary = None
    if summary is None:
        summary = {
            "actions": [],
            "queries": [],
            "explanations": [],
            "decisions": []
        }

    pattern = f"{input_dir.rstrip('/')}/*.json"
    # Gather JSON files to analyze
    filepaths = sorted(glob.glob(pattern))
    if not filepaths:
        print(f"No JSON files found in {input_dir}", file=sys.stderr)
        return
    print(f"Found {len(filepaths)} JSON files in {input_dir}")
    # Iterate over each optimized JSON file
    for idx, filepath in enumerate(filepaths, start=1):
        print(f"Analyzing file {idx}/{len(filepaths)}: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Determine list of chunks to analyze: support both direct chunks and optimizer output
        examples = []
        if isinstance(data, dict) and "training_examples" in data:
            # Optimizer output: analyze each example's segment
            for ex in data.get("training_examples", []):
                # each ex is a dict with 'segment' or 'content'
                examples.append(ex)
        else:
            # Single raw or cleaned chunk
            examples.append(data)
        # Analyze each chunk/example
        for ex in examples:
            text = ex.get("segment") or ex.get("content") or json.dumps(ex)
            result = analyze_chunk(text, model)
            # Aggregate into summary
            for key in summary:
                summary[key].extend(result.get(key, []))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Print summary of results
    print(f"Summary written to {output_file}:")
    print(f"  actions: {len(summary['actions'])}")
    print(f"  queries: {len(summary['queries'])}")
    print(f"  explanations: {len(summary['explanations'])}")
    print(f"  decisions: {len(summary['decisions'])}")

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
        help="Output JSON summary file or directory to write timestamped summaries."
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help="OpenAI model to use for analysis."
    )
    parser.add_argument(
        "--append", action="store_true",
        help="If set and output exists, load and append to existing summary."
    )
    args = parser.parse_args()
    # Attach append flag to main
    setattr(main, 'append', args.append)
    main(args.input, args.output, args.model)