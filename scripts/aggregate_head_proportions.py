# scripts/aggregate_head_proportions.py
"""
Aggregate head proportions from multiple JSON files by averaging.

Usage:
    python scripts/aggregate_head_proportions.py [directory]

    directory: Directory containing head proportion JSON files (default: logs/head_proportions/rms_t0.02)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Default directory containing head proportion results
DEFAULT_DIR = "logs/head_proportions/rms_t0.02"


def main(directory=None):
    if directory is None:
        directory = DEFAULT_DIR

    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    # Find all JSON files (excluding the output file)
    json_files = [f for f in dir_path.glob("*.json") if f.name != "overall_head_proportions.json"]
    print(f"Found {len(json_files)} JSON files in {directory}")

    if not json_files:
        print("No JSON files found")
        return

    # Collect all values for each head
    head_values = defaultdict(list)

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            print(f"Processing {json_file.name}...")
            for head, value in data.items():
                head_values[head].append(value)

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    # Average the values
    averaged = {}
    for head, values in sorted(head_values.items()):
        averaged[head] = sum(values) / len(values)

    # Write output
    output_file = dir_path / "overall_head_proportions.json"
    with open(output_file, 'w') as f:
        json.dump(averaged, f, indent=2)

    print(f"\nAveraged {len(json_files)} files into {output_file}")
    print(f"Total heads: {len(averaged)}")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    main(directory)
