import json
from pathlib import Path

def save_triplets(idx, triplets, output_text, output_dir):
    """Save the cleaned triplets and raw LLM output."""
    Path(f"{output_dir}/logs").mkdir(parents=True, exist_ok=True)

    # Save cleaned triplets
    with open(f"{output_dir}/triplets_chunk_{idx+1}.json", "w", encoding="utf-8") as f:
        json.dump(triplets, f, indent=2)

    # Save raw output
    with open(f"{output_dir}/logs/output_raw_triplets_chunk_{idx+1}.txt", "w", encoding="utf-8") as f:
        f.write(output_text)


def save_error(idx, output_text, output_dir):
    """Save LLM output when JSON repair fails."""
    Path(f"{output_dir}/logs").mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/logs/error_triplets_chunk_{idx+1}.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
