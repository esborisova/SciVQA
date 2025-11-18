import argparse
import json
import re
import os
from ..utils.process_tex_source import (
    remove_author_title,
    remove_commands,
    remove_bibliography_paragraphs,
    remove_command_only_paragraphs,
    remove_empty_sections,
    has_real_content,
)

STRUCTURAL_COMMANDS = {
    "maketitle",
    "tableofcontents",
    "newpage",
    "clearpage",
    "label",
    "PACS",
    "printaddresses",
    "author",
    "title",
    "vspace",
    "hspace",
    "smallskip",
    "medskip",
    "bigskip",
    "clearpage",
    "footnotesize",
    "normalsize",
    "appendices",
    "elecappendix",
    "begin",
    "end",
    "section",
    "subsection",
    "subsubsection",
    "appendix",
    "pinch",
    "indent",
    "noindent",
    "input",
    "large",
    "pagebreak",
    "clubpenalty",
    "widowpenalty",
    "bibliography",
    "bibliographystyle",
    "nocite",
    "bibitem",
    "thebibliography",
    "notice",
    "usepackage",
    "renewcommand",
    "newcommand",
    "documentclass",
    "pagestyle",
    "thispagestyle",
    "vspace",
    "hspace",
    "noindent",
    "centering",
    "clearpage",
    "pagebreak",
    "linebreak",
    "flushleft",
    "flushright",
    "catchline",
}

SECTION_COMMANDS = {"section", "subsection", "subsubsection", "appendix", "title"}

header_patterns = [
    r"\\title\s*\{.*?\}",
    r"\\icmltitle\s*\{.*?\}",
    r"\\titlerunning\s*\{.*?\}",
    r"\\author\s*\{.*?\}",
    r"\\icmlauthor\s*\{.*?\}",
    r"\\correspondingauthor\s*\{.*?\}",
    r"\\icmlcorrespondingauthor\s*\{.*?\}",
    r"\\authorrunning\s*\{.*?\}",
    r"\\editor\s*\{.*?\}",
    r"\\date\s*\{.*?\}",
    r"\\affiliation\s*\{.*?\}",
    r"\\icmlaffiliation\s*\{.*?\}",
    r"\\address\s*\{.*?\}",
    r"\\institute\s*\{.*?\}",
    r"\\email\s*\{.*?\}",
    r"\\thanks\s*\{.*?\}",
    r"\\maketitle",
    r"\\and",
    r"\\authorinfo\s*\{.*?\}",
]


combined_pattern = re.compile(
    "(" + "|".join(header_patterns) + ")",
    flags=re.DOTALL,
)

figure_table_patterns = [
    r"\\begin\s*\{tikzpicture\}.*?\\end\s*\{tikzpicture\}",
    r"\\begin\s*\{wrapfigure\}.*?\\end\s*\{wrapfigure\}",
    r"\\begin\s*\{subfigure\}.*?\\end\s*\{subfigure\}",
    r"\\caption\s*\{.*?\}",
    r"\\captionsetup\s*\{.*?\}",
    r"\\begin{picture}.*?\\end\s*\{picture\}",
]

figure_table_pattern = re.compile(
    "(" + "|".join(figure_table_patterns) + ")",
    flags=re.DOTALL,
)


def main(args):
    root_dir = args.tex_folder
    save_path = args.output_folder

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            json_path = os.path.join(dirpath, filename)
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {json_path}")
                    continue

            for sec in data["sections"]:
                remove_author_title(sec, combined_pattern)
                remove_commands(sec, figure_table_pattern)
                remove_bibliography_paragraphs(sec)
                remove_command_only_paragraphs(
                    sec, SECTION_COMMANDS, STRUCTURAL_COMMANDS
                )
            data["sections"] = [
                remove_empty_sections(sec)
                for sec in data["sections"]
                if has_real_content(sec)
            ]

            relative_path = os.path.relpath(dirpath, root_dir)
            save_dir = os.path.join(save_path, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, filename)
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process LaTeX .tar.gz files and extract structure JSONs."
    )
    parser.add_argument(
        "--tex_folder",
        required=True,
        help="Path to folder containing .tar or .tar.gz files",
    )
    parser.add_argument(
        "--output_folder", required=True, help="Folder to save parsed JSON files"
    )
    args = parser.parse_args()
    main(args)
