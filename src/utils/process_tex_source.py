"""Functions for parsing and cleaning LaTeX source of papers."""

import re
import os
import tarfile


# ------Functions for parsing------
def split_paragraphs(text: str) -> list:
    """Split text into paragraphs based on newlines."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def strip_comments(tex_text: str) -> str:
    """Remove LaTeX comments (everything after % unless escaped) and comment environments."""
    tex_text = re.sub(
        r"\\begin\{comment\}.*?\\end\{comment\}", "", tex_text, flags=re.DOTALL
    )
    tex_text = re.sub(r"(?<!\\)%.*", "", tex_text)
    return tex_text


def process_plain_text(
    accumulated_text: list, env_stack: list, output_blocks: list
) -> None:
    """Process accumulated plain text outside any environment into paragraphs."""
    if accumulated_text and not env_stack:
        prose = "".join(accumulated_text)
        output_blocks.extend(split_paragraphs(prose))
        accumulated_text.clear()


def _is_excluded_env(env_block: str, exclude_envs: set) -> bool:
    return any(re.search(rf"\\begin\s*\{{{e}\*?\}}", env_block) for e in exclude_envs)


def process_env_block(
    accumulated_env: list, exclude_envs: set, output_blocks: list
) -> None:
    """Process a full LaTeX environment block and add to output if not excluded."""
    env_block = "".join(accumulated_env).strip()
    if not _is_excluded_env(env_block, exclude_envs):
        output_blocks.append(env_block)
    accumulated_env.clear()


def _is_env_begin(token: str) -> bool:
    return re.match(r"\\begin\s*\{(.*?)\}", token)


def _is_env_end(token: str) -> bool:
    return re.match(r"\\end\s*\{(.*?)\}", token)


def extract_env_name(token: str) -> str:
    match = re.match(r"\\(?:begin|end)\s*\{(.*?)\}", token)
    return match.group(1) if match else ""


def extract_blocks(text: str, exclude_envs: set) -> list:
    """Split LaTeX text into prose and non-excluded environments."""
    tokens = re.split(r"(\\begin\s*\{.*?\}|\\end\s*\{.*?\})", text, flags=re.DOTALL)
    blocks, buffer, stack = [], [], []

    for token in filter(str.strip, tokens):
        if _is_env_begin(token):
            process_plain_text(buffer, stack, blocks)
            stack.append(extract_env_name(token))
            buffer.append(token)
        elif _is_env_end(token):
            buffer.append(token)
            if stack:
                stack.pop()
                if not stack:
                    process_env_block(buffer, exclude_envs, blocks)
        else:
            buffer.append(token)

    process_plain_text(buffer, stack, blocks)
    return [block for block in blocks if block.strip()]


def find_section_matches(text: str, level: str):
    return re.finditer(rf"\\{level}\*?\s*\{{", text, flags=re.DOTALL)


def get_section_body(text: str, index: int, matches, title_end) -> str:
    """Return the section body text between current section and the next."""
    start = title_end
    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
    return text[start:end]


def find_brace_content(text: str, start_index: int) -> tuple:
    """Find the content inside curly braces."""
    while start_index < len(text) and text[start_index] != "{":
        start_index += 1
    if start_index >= len(text):
        return "", start_index

    depth = 0
    content = []
    for i in range(start_index, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                content.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(content), i + 1
            content.append(ch)
        elif depth > 0:
            content.append(ch)
    return "".join(content), len(text)


def split_section_results(results: list) -> tuple:
    """Separate content from subsection dicts."""
    if isinstance(results, list):
        content = [block for block in results if isinstance(block, str)]
        subsections = [block for block in results if isinstance(block, dict)]
        return content, subsections
    return [], []


def compile_lower_level_regex(levels: list):
    pattern = "|".join([rf"\\{level}\*?\s*\{{" for level in levels])
    return re.compile(pattern, flags=re.DOTALL)


def make_section(title: str, content: list, subsections: list) -> dict:
    return {"title": title, "content": content, "subsections": subsections}


def extract_subsection(
    body: str,
    match: re.Match,
    lower_re: re.Pattern,
    exclude_envs: set,
    lower_levels: list,
) -> tuple:
    """Extract title, content, and subsections for a single lower-level heading."""
    lower_title, lower_end = find_brace_content(body, match.end() - 1)
    next_match = lower_re.search(body, lower_end)
    lower_body = body[lower_end : (next_match.start() if next_match else len(body))]
    lower_recursive = extract_sections_recursive(lower_body, exclude_envs, lower_levels)
    lower_content, lower_subs = split_section_results(lower_recursive)
    subsection = make_section(lower_title.strip(), lower_content, lower_subs)
    return subsection, next_match.start() if next_match else len(body)


def split_into_subsections(body, exclude_envs: set, levels: list) -> tuple:
    """Separate current-level content and recursively extracted subsections."""
    if len(levels) == 1:
        return [], extract_blocks(body, exclude_envs)

    subsections = []
    content_blocks = []
    lower_levels = levels[1:]
    lower_re = compile_lower_level_regex(lower_levels)

    pos = 0
    for match in lower_re.finditer(body):
        if match.start() > pos:
            content_blocks.extend(
                extract_blocks(body[pos : match.start()], exclude_envs)
            )
        subsection, pos = extract_subsection(
            body, match, lower_re, exclude_envs, lower_levels
        )
        subsections.append(subsection)
    if pos < len(body):
        content_blocks.extend(extract_blocks(body[pos:], exclude_envs))
    return subsections, content_blocks


def extract_sections_recursive(tex_text: str, exclude_envs: set, levels: list) -> list:
    """Recursively extract sections/subsections into structured dicts."""
    if not levels:
        return extract_blocks(tex_text, exclude_envs)

    level = levels[0]
    matches = list(find_section_matches(tex_text, level))
    if not matches:
        return extract_blocks(tex_text, exclude_envs)

    sections = []
    for i, match in enumerate(matches):
        title, title_end = find_brace_content(tex_text, match.end() - 1)
        body = get_section_body(tex_text, i, matches, title_end)
        subsections, content = split_into_subsections(body, exclude_envs, levels)
        cont_dict = make_section(title.strip(), content, subsections)
        sections.append(cont_dict)
    return sections


def extract_env_abstract(tex_text: str, exclude_envs: set):
    """Handle \begin{abstract}...\end{abstract}."""
    match = re.search(
        r"\\begin\s*\{abstract\}(.*?)\\end\s*\{abstract\}", tex_text, re.DOTALL
    )
    if match:
        abstract_body = match.group(1)
        tex_text = tex_text.replace(match.group(0), "")
        content = extract_blocks(abstract_body, exclude_envs)
        return tex_text.strip(), make_section("Abstract", content, [])
    return None, None


def extract_macro_abstract(tex_text: str, exclude_envs: set):
    """Handle \abstract{...}."""
    match = re.search(r"\\abstract\s*\{", tex_text)
    if match:
        abstract_body, end_idx = find_brace_content(tex_text, match.end() - 1)
        tex_text = tex_text[: match.start()] + tex_text[end_idx:]
        content = extract_blocks(abstract_body, exclude_envs)
        return tex_text.strip(), make_section("Abstract", content, [])
    return None, None


def extract_abstract(tex_text: str, exclude_envs: set) -> tuple:
    """Extract abstract defined as either \begin{abstract}...\end{abstract}
    or \abstract{}.
    """
    extractors = [extract_env_abstract, extract_macro_abstract]
    for extractor in extractors:
        new_text, section = extractor(tex_text, exclude_envs)
        if section:
            return new_text, section
    return tex_text, None


def parse_latex_structure(tex_text: str, exclude_envs: set, levels: list):

    tex_text = strip_comments(tex_text)
    result = {"sections": []}

    doc_match = re.search(r"\\begin\s*\{document\}(.*)", tex_text, re.DOTALL)

    if doc_match:
        tex_text = doc_match.group(1)

    tex_text, abs_section = extract_abstract(tex_text, exclude_envs)
    if abs_section:
        result["sections"].append(abs_section)

    tex_text = tex_text.strip()

    first_heading = re.search(r"\\(section|subsection)\*?\s*\{", tex_text)
    pre_text = (
        tex_text[: first_heading.start()].strip() if first_heading else tex_text.strip()
    )
    after_headings = tex_text[first_heading.start() :] if first_heading else ""

    if not pre_text and not after_headings:
        return result

    if first_heading:
        if first_heading.group(1) == "section":
            if pre_text:
                result["sections"].append(
                    make_section("", extract_blocks(pre_text, exclude_envs), [])
                )
            sections = extract_sections_recursive(after_headings, exclude_envs, levels)
            result["sections"].extend(sections if isinstance(sections, list) else [])
        else:
            content = extract_blocks(pre_text, exclude_envs) if pre_text else []
            subsections = extract_sections_recursive(tex_text, exclude_envs, levels[1:])
            result["sections"].append(
                make_section(
                    "", content, subsections if isinstance(subsections, list) else []
                )
            )
    else:
        if pre_text:
            result["sections"].append(
                make_section("", extract_blocks(pre_text, exclude_envs), [])
            )
    return result


def find_main_tex_in_tar(tar_path: str):
    """Find main TeX files inside a tar.gz without extracting all."""
    tex_files = []
    begin_doc_re = re.compile(r"\\begin\s*\{\s*document\s*\}", re.IGNORECASE)

    try:
        with tarfile.open(tar_path, "r:*") as tar_ref:
            for member in tar_ref.getmembers():
                if member.isfile() and member.name.endswith(".tex"):
                    file = tar_ref.extractfile(member)
                    if file is not None:
                        content = file.read().decode("utf-8", errors="ignore")
                        if begin_doc_re.search(content):
                            tex_files.append(member.name)
    except (tarfile.ReadError, tarfile.CompressionError, EOFError) as e:
        return [], str(e)

    return list(set(tex_files)), ""


def build_tar_index(tar_ref):
    index = {}
    for member in tar_ref.getmembers():
        if not member.isfile():
            continue
        base = os.path.basename(member.name)
        index.setdefault(base, []).append(member.name)
    return index


def read_tex_from_tar(tex_name: str, tar_ref):
    try:
        member = tar_ref.getmember(tex_name)
    except KeyError:
        return None
    file_obj = tar_ref.extractfile(member)
    if not file_obj:
        return None
    return file_obj.read().decode("utf-8", errors="ignore")


def resolve_tex_ref(base_tex_name, ref_path: str):
    """
    Resolve a LaTeX reference paths.
    """
    if os.path.isabs(ref_path):
        return os.path.normpath(ref_path)
    base_dir = os.path.dirname(base_tex_name)
    top_level_prefix = base_dir.split(os.sep, 1)[0]
    if ref_path.startswith(top_level_prefix + os.sep):
        return os.path.normpath(ref_path)
    return os.path.normpath(os.path.join(base_dir, ref_path))


def tex_candidates(path, base_tex_name):
    has_tex_ext = path.endswith(".tex")
    base = resolve_tex_ref(base_tex_name, path)
    cands = []
    root_relative = os.path.normpath(path)
    cands.extend(
        [root_relative + ".tex", root_relative] if not has_tex_ext else [root_relative]
    )
    cands.extend([base + ".tex", base] if not has_tex_ext else [base])
    return cands

BRACED_REF_RE = re.compile(r"\\(?:input|include|subfile|latexfile)\s*\{([^\}]+)\}")
NO_BRACE_REF_RE = re.compile(
        r"\\(?:input|include|subfile|latexfile)\s+([^\s\{\}]+)"
    )
IMPORT_REF_RE = re.compile(r"\\(?:import|subimport)\s*\{([^\}]+)\}\s*\{([^\}]+)\}")

def extract_tex_ref(tex_content: str):
    """
    Extract all \input / \include / \import / \subimport references,
    ignoring any that appear before \begin{document}.
    """

    begin_match = re.search(
        r"\\begin\s*\{\s*document\s*\}", tex_content, flags=re.IGNORECASE
    )
    doc_start = begin_match.end() if begin_match else 0

    matches = []

    for match in IMPORT_REF_RE.finditer(tex_content):
        if match.start() >= doc_start:
            dir_part, file_part = match.group(1), match.group(2)
            full_path = os.path.normpath(os.path.join(dir_part, file_part))
            matches.append((match.start(), match.end(), full_path))

    for match in BRACED_REF_RE.finditer(tex_content):
        if match.start() >= doc_start:
            matches.append((match.start(), match.end(), match.group(1)))

    for match in NO_BRACE_REF_RE.finditer(tex_content):
        if match.start() >= doc_start:
            matches.append((match.start(), match.end(), match.group(1)))

    matches.sort(key=lambda x: x[0])
    return matches


def find_tex_reference(ref_path, base_tex_name, tar_ref, tar_index):
    """
    Try to locate a .tex file in the tar archive based on the given reference path.
    Returns the resolved path inside the tar if found, else None.
    Includes a fallback for quoted paths.
    """
    for cand in tex_candidates(ref_path, base_tex_name):
        try:
            tar_ref.getmember(cand)
            return cand
        except KeyError:
            continue
    fname = os.path.basename(
        ref_path if ref_path.endswith(".tex") else ref_path + ".tex"
    )
    if fname in tar_index:
        for alt_path in tar_index[fname]:
            if alt_path.endswith(fname):
                return alt_path
    if '"' in ref_path or "'" in ref_path:
        stripped = ref_path.strip().strip('"').strip("'")
        if stripped != ref_path:
            return find_tex_reference(stripped, base_tex_name, tar_ref, tar_index)
    return None

EXCLUDE_ENVS = {
        "table",
        "table*",
        "tabular",
        "tabular*",
        "figure",
        "figure*",
        "longtable",
        "sidewaystable",
        "wrapfigure",
    }

LEVELS = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]

def parse_tex_file(
    tex_name, tar_ref, processed_files=None, tar_path=None, tar_index=None, errors=None
):
    """
    Recursively parse a .tex file in a tar archive and its \input / \include references.
    Returns a merged structure (list of sections).
    """

    if processed_files is None:
        processed_files = set()
    if tar_index is None:
        tar_index = build_tar_index(tar_ref)
    if errors is None:
        errors = []

    if tex_name in processed_files:
        return []
    processed_files.add(tex_name)

    tex_content = read_tex_from_tar(tex_name, tar_ref)
    if tex_content is None:
        errors.append(
            {"file": tar_path, "stage": "parse_tex_file", "missing_tex": tex_name}
        )
        return []

    tex_content = strip_comments(tex_content)
    final_structure = []
    matches = extract_tex_ref(tex_content)
    pos = 0

    for start, end, ref_path in matches:
        snippet = tex_content[pos:start]
        final_structure.extend(
            parse_latex_structure(snippet, EXCLUDE_ENVS, LEVELS)["sections"]
        )

        found_path = find_tex_reference(ref_path, tex_name, tar_ref, tar_index)
        if found_path:
            final_structure.extend(
                parse_tex_file(
                    found_path,
                    tar_ref,
                    processed_files,
                    tar_path=tar_path,
                    tar_index=tar_index,
                    errors=errors,
                )
            )
        else:
            errors.append(
                {"file": tar_path, "stage": "parse_tex_file", "missing_tex": ref_path}
            )
        pos = end

    final_structure.extend(
        parse_latex_structure(tex_content[pos:], EXCLUDE_ENVS, LEVELS)["sections"]
    )
    return final_structure


# ------Functions for cleaning------


def remove_author_title(section, header_pattern):
    section["content"] = [
        p for p in section.get("content", []) if not header_pattern.search(p)
    ]

    for sub in section.get("subsections", []):
        remove_author_title(sub)

    return section


def remove_commands(section, header_pattern, figure_table_pattern):
    section["content"] = [
        p for p in section.get("content", []) if not figure_table_pattern.search(p)
    ]
    for sub in section.get("subsections", []):
        remove_author_title(sub, header_pattern)

    return section


def is_structural_only(paragraph, section_commands, structural_commands):
    """
    Return True if paragraph contains only structural LaTeX commands with no meaningful text.
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return True

    text_outside = re.sub(r"\\\s*[a-zA-Z]+(\[[^\]]*\])?\s*(\{[^{}]*\})?", "", paragraph)
    text_outside = re.sub(r"[\\%$&_^\~{}]", "", text_outside).strip()
    if text_outside:
        return False

    commands = re.findall(r"\\\s*([a-zA-Z]+)(\[[^\]]*\])?\s*(\{([^{}]*)\})?", paragraph)

    for cmd, _, _, arg in commands:
        if cmd in section_commands:
            if arg and arg.strip():
                return False
        elif cmd not in structural_commands:
            return False
    return True


def remove_command_only_paragraphs(section, section_commands, structural_commands):
    """Recursively remove paragraphs that are only structural/technical commands."""
    section["content"] = [
        p
        for p in section.get("content", [])
        if not is_structural_only(p, section_commands, structural_commands)
    ]
    for sub in section.get("subsections", []):
        remove_command_only_paragraphs(sub)

    return section


def has_real_content(section):
    """Return True if this section has any real text, title, or non-empty subsections."""
    if section.get("title", "").strip():
        return True
    if any(p.strip() for p in section.get("content", [])):
        return True
    if any(has_real_content(sub) for sub in section.get("subsections", [])):
        return True
    return False


def remove_empty_sections(section):
    """Recursively remove empty subsections and return cleaned section."""
    cleaned_subs = []
    for sub in section.get("subsections", []):
        cleaned_sub = remove_empty_sections(sub)
        if has_real_content(cleaned_sub):
            cleaned_subs.append(cleaned_sub)
    section["subsections"] = cleaned_subs
    return section


bibliography_env_pattern = re.compile(
    r"\\begin\s*\{thebibliography\}.*?\\end\s*\{thebibliography\}",
    flags=re.DOTALL,
)

bib_commands = [
    r"^\\bibitem",
    r"^\\providecommand\s*\{\\natexlab",
    r"^\\begin\s*\{thebibliography\}",
    r"\\end\s*\{thebibliography\}",
]

bib_commands_pattern = re.compile(
    "(" + "|".join(bib_commands) + ")",
    flags=re.MULTILINE,
)


def remove_bibliography_paragraphs(section):
    """
    Removes paragraphs that contain bibliography environments or individual bib entries.
    Works even if bibliography is split across multiple paragraphs.
    """

    new_content = []
    for p in section.get("content", []):
        if bibliography_env_pattern.search(p):
            continue
        if bib_commands_pattern.search(p):
            continue
        new_content.append(p)
    section["content"] = new_content
    for sub in section.get("subsections", []):
        remove_bibliography_paragraphs(sub)
    return section
