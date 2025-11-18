"""Add unique IDs to paragraphs in JSON files per paper."""

import json
import pandas as pd
from pathlib import Path
import argparse
from ..utils.data_preprocessing import add_hash_to_sections


def main(args):

    root_dir = Path(args.root_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for paper_folder in root_dir.iterdir():
        if not paper_folder.is_dir():
            continue

        json_files = list(paper_folder.rglob("*.json"))

        if not json_files:
            continue

        for json_file in json_files:
            json_file = Path(json_file)
            if not json_file.exists():
                continue

            with open(json_file, "r", encoding="utf-8") as f:
                paper = json.load(f)

            add_hash_to_sections(paper.get("sections", []), str(paper_folder))

            out_folder = output_dir / paper_folder.name
            out_folder.mkdir(parents=True, exist_ok=True)
            out_file = out_folder / json_file.name
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(paper, f, indent=2, ensure_ascii=False)

            print(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_folder",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        required=True,
    )
    args = parser.parse_args()
    main(args)
