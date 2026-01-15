"""Script for parsing markdown files into sections and paragraphs."""

import os
import json
import argparse
import pandas as pd
from ..utils.data_preprocessing import (
    parse_markdown,
    remove_captions,
    clean_md,
    drop_sections_before_abstract,
    build_section_json,
)

def main(args):

    data_path = args.data_path
    input_path = args.input_path
    output_path = args.output_path
    threshold = float(args.sim_threshold)
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_pickle(data_path)
    paper_ids = list(set(df["paper_id"].tolist()))
    captions = df["caption"].tolist()
    captions = [cap.lower() for cap in captions]

    missing_abstracts = []
    for paper_id in paper_ids:
        folder_path = os.path.join(input_path, paper_id)
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(".md"):
                    md_path = os.path.join(root, filename)
                    with open(md_path, "r", encoding="utf-8") as f:
                        markdown_text = f.read()
                    markdown_text = clean_md(markdown_text)
                    markdown_text = remove_captions(markdown_text, captions, threshold)
                    parsed = parse_markdown(markdown_text)
                    # remove sections with references
                    parsed = [
                        sec
                        for sec in parsed
                        if sec.get("title", "").strip().lower() != "references"
                    ]
                    # remove sections with title and authors
                    parsed, has_abstract = drop_sections_before_abstract(parsed)
                    if not has_abstract:
                        missing_abstracts.append(
                            {"paper_id": paper_id, "md_path": md_path}
                        )
                    parsed_sections = build_section_json(parsed, md_path)
                    output = {"sections": parsed_sections}
                    json_filename = os.path.splitext(filename)[0] + ".json"
                    json_path = os.path.join(output_path, json_filename)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(output, f, ensure_ascii=False, indent=2)
                    print(f"Parsed {filename} into {json_filename}")

    df_missing = pd.DataFrame(missing_abstracts)
    csv_path = os.path.join(output_path, "missing_abstracts.csv")
    df_missing.to_csv(csv_path, index=False)
    print(f"Saved {len(df_missing)} missing abstracts to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the pkl file with already collected images",
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sim_threshold", help="Threshold for fuzzy match")
    args = parser.parse_args()
    main(args)
