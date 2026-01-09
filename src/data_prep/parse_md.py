"""Script for parsing markdown files into sections and paragraphs."""

import re
import os
import json
import argparse
import pandas as pd
from bs4 import BeautifulSoup


def is_header(line):
    match = re.match(r"^(#+)\s*(.*)", line)
    return match.group(2) if match else None


def add_paragraph(current_section, paragraph_lines):
    if paragraph_lines:
        paragraph = " ".join(paragraph_lines)
        current_section["paragraphs"].append(paragraph)
        paragraph_lines.clear()


def parse_markdown(md_text: str) -> list:
    sections = []
    current_section = {"title": "", "paragraphs": []}
    paragraph_lines = []

    for line in md_text.splitlines():
        line = line.strip()
        if not line:
            add_paragraph(current_section, paragraph_lines)
            continue

        header_text = is_header(line)
        if header_text is not None:
            add_paragraph(current_section, paragraph_lines)
            if current_section["title"] or current_section["paragraphs"]:
                sections.append(current_section)
            current_section = {"title": header_text, "paragraphs": []}
        else:
            paragraph_lines.append(line)

    add_paragraph(current_section, paragraph_lines)
    if current_section["title"] or current_section["paragraphs"]:
        sections.append(current_section)
    return sections

def remove_tables(md_text):
    soup = BeautifulSoup(md_text, "html.parser")
    for table in soup.find_all("table"):
        table.decompose()
    return soup.get_text()

def clean_md(md_text):
    cleaned_text = re.sub(r"<!--.*?-->", "", md_text, flags=re.DOTALL)
    #remove images
    cleaned_text = re.sub(r"^!\[.*?\]\(.*?\)\s*$", "", cleaned_text , flags=re.MULTILINE)
    cleaned_text = remove_tables(cleaned_text)
    return cleaned_text


def main(args):

    data_path = args.data_path
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_pickle(data_path)
    paper_ids = df['paper_id'].tolist()

    for paper_id in paper_ids:
        folder_path = os.path.join(input_path, paper_id)
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(".md"):
                    md_path = os.path.join(root, filename)
                    with open(md_path, "r", encoding="utf-8") as f:
                        markdown_text = f.read()
                    markdown_text = clean_md(markdown_text)
                    parsed = parse_markdown(markdown_text)

                    json_filename = os.path.splitext(filename)[0] + ".json"
                    json_path = os.path.join(output_path, json_filename)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)

                    print(f"Parsed {filename} into {json_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
