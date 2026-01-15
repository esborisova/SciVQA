from typing import List
import os
import requests
import re
import json
import hashlib
import pandas as pd
import uuid
from rapidfuzz import fuzz, process
from bs4 import BeautifulSoup

def download_file(url: str, filename: str) -> tuple:
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return f"{response.status_code}, ok", ""
        elif response.status_code == 404:
            return f"{response.status_code}, missing", "404 not found"
        else:
            return "error", f"HTTP {response.status_code}"
    except Exception as e:
        return "error", str(e)


def extract_paper_id(file_name: str) -> str:
    paper_id = file_name.split("figures_meta_")[-1].split(".json")[0]
    paper_id = re.sub(r"v\d+$", "", paper_id)
    return paper_id


def transform_filename(fname: str) -> str:
    """
    E.g., C00-1065-Figure2-1.png -> C00-1065.pdf-Figure2.png
    """
    if fname.endswith("-1.png"):
        fname = fname.replace("-1.png", ".png")
    if "-Figure" in fname:
        fname = fname.replace("-Figure", ".pdf-Figure", 1)
    return fname


def match_figures(row, images: List[str], captions: List[str]) -> str:
    fname = row["image_file"]
    transformed = transform_filename(fname)

    if (
        fname in images
        or transformed in images
        or (fname in images and row["caption"] in captions)
    ):
        return "drop"
    return "keep"


def extract_meta(folder_path: str) -> List:
    records = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            json_file = os.path.join(folder_path, file)
            with open(json_file, "r") as f:
                metadata = json.load(f)
        for entry in metadata:
            render_url = entry.get("renderURL", "")
            if render_url:
                file_name = os.path.basename(render_url)
                caption = entry.get("caption", "")
                records.append(
                    {"file_name": file_name, "caption": caption, "source_json": file}
                )
    return records


def get_paragraph_hash(paper_path: str, section_title: str, paragraph_text: str) -> str:
    combined = f"{paper_path}_{section_title}_{paragraph_text}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def add_hash_to_sections(sections: list, paper_path: str):
    for section in sections:
        new_content = []
        for para in section.get("content", []):
            para_hash = get_paragraph_hash(paper_path, section.get("title", ""), para)
            new_content.append({"id": para_hash, "paragraph": para})
        section["content"] = new_content
        if "subsections" in section:
            add_hash_to_sections(section["subsections"], paper_path)


def combine_id_versions(row):
    id_ = row["id"]
    versions = row["versions"]
    combined = []
    if isinstance(versions, list):
        for v in versions:
            if isinstance(v, dict) and "version" in v:
                combined.append(id_ + v["version"])
    return combined


def generate_unique_id(existing_ids):
    while True:
        new_id = uuid.uuid4().hex
        if new_id not in existing_ids:
            existing_ids.add(new_id)
            return new_id


def extract_main_label(text: str):
    """
    Extract the main figure/table label, e.g.,
    Figure 2(a) -> Figure2, Table 1.1(b) -> Table1.1,
    TABLE I -> TableI
    """
    if pd.isna(text) or not isinstance(text, str):
        return None

    text = text.strip()

    pattern = r"\b(Figure|Fig\.?|Table)\s*[\.:]?\s*([0-9]+(?:\.[0-9]+)*|[IVXLCDM]+)\s*(?:\(?[a-z]\)?)?"
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        label_type = match.group(1)
        label_num = match.group(2)
        if label_type.lower().startswith("fig"):
            label_type = "Figure"
        else:
            label_type = "Table"
        return f"{label_type}{label_num}"

    return None


def add_filenames(df: pd.DataFrame):
    filenames = []

    paper_counters = {}

    for _, row in df.iterrows():
        paper_id = row.get("paper_id")
        if not paper_id:
            filenames.append(None)
            continue

        label = extract_main_label(row.get("caption"))
        if not label:
            label = extract_main_label(row.get("footnote"))
        if not label:
            filenames.append(None)
            continue

        paper_counters.setdefault(paper_id, {})
        paper_counters[paper_id].setdefault(label, 0)

        paper_counters[paper_id][label] += 1
        sub_index = paper_counters[paper_id][label]

        filename = f"{paper_id}-{label}-{sub_index}.png"
        filenames.append(filename)

    return pd.Series(filenames)


def extract_title_prefix(text: str):
    if not text:
        return None
    text = text.lower()
    m = re.match(r"(table|figure|fig)\s*[\.:]?\s*([0-9]+[a-z]?(?:\([a-z]\))?)", text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return None


def is_header(line: str):
    match = re.match(r"^(#+)\s*(.*)", line)
    return match.group(2) if match else None


def add_paragraph(current_section: dict, paragraph_lines: list):
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


def remove_tables(md_text: str) -> str:
    soup = BeautifulSoup(md_text, "html.parser")
    for table in soup.find_all("table"):
        table.decompose()
    return soup.get_text()


def remove_captions(md_text: str, captions: list, threshold: float = 80) -> str:
    md_text = md_text.replace("\r\n", "\n")
    paragraphs = re.split(r"\n\s*\n", md_text)
    kept_paragraphs = []
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        match = process.extractOne(
            stripped.lower(),
            captions,
            scorer=fuzz.token_sort_ratio,
        )
        if match is None or match[1] < threshold:
            kept_paragraphs.append(para)
    cleaned_md = "\n\n".join(kept_paragraphs)
    return cleaned_md


def clean_md(md_text: str) -> str:
    cleaned_text = re.sub(r"<!--.*?-->", "", md_text, flags=re.DOTALL)
    # remove images
    cleaned_text = re.sub(r"^!\[.*?\]\(.*?\)\s*$", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = remove_tables(cleaned_text)
    return cleaned_text


def make_abstract_regex(N: int = 5) -> re.Pattern:
    return re.compile(rf"^.{{0,{N}}}abstract.{{0,{N}}}$", re.IGNORECASE)


def is_abstract_title(title: str, N: int) -> bool:
    ABSTRACT_RE = make_abstract_regex(N=N)
    title = title.strip()
    return len(title) <= 20 and bool(ABSTRACT_RE.match(title.lower()))


def drop_sections_before_abstract(parsed_sections: list, N: int) -> tuple:
    for i, sec in enumerate(parsed_sections):
        if is_abstract_title(sec.get("title", ""), N=N):
            return parsed_sections[i:], True
    return parsed_sections, False


def build_section_json(parsed_sections: list, paper_path: str) -> list:
    sections = []
    for sec in parsed_sections:
        content = []
        for para in sec.get("paragraphs", []):
            para = para.strip()
            # clean from empty paragraphs
            if not para:
                continue
            para_hash = get_paragraph_hash(paper_path, sec.get("title", ""), para)
            content.append({"id": para_hash, "paragraph": para})
        # clean from empty sections
        if not content and not sec.get("title", "").strip():
            continue
        sections.append(
            {"title": sec.get("title", ""), "content": content, "subsections": []}
        )
    return sections
