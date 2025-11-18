"""Parse LaTeX files from arXiv and save into structured JSON format."""

import os
import json
import tarfile
from datetime import datetime
import argparse
from ..utils.process_tex_source import find_main_tex_in_tar, parse_tex_file


def main(args):

    tex_folder = args.tex_folder
    files = sorted(f for f in os.listdir(tex_folder) if f.endswith((".tar.gz", ".tar")))

    errors = []
    for fname in files:
        tar_path = os.path.join(tex_folder, fname)
        main_tex_files, err = find_main_tex_in_tar(tar_path)

        if err:
            errors.append(
                {"file": fname, "stage": "find_main_tex_in_tar", "error": err}
            )
            continue

        if not main_tex_files:
            errors.append(
                {
                    "file": fname,
                    "stage": "find_main_tex_in_tar",
                    "error": "No main TeX file found",
                }
            )
            continue
        try:
            with tarfile.open(tar_path, "r:*") as tar_ref:
                for tex_file in main_tex_files:
                    print(f"Processing {tar_path}")
                    structure_sections = parse_tex_file(
                        tex_file, tar_ref, tar_path=tar_path, errors=errors
                    )
                    print(f"Processed {tar_path}")
                    result_json = {"sections": structure_sections}
                    relative_path = os.path.splitext(tex_file)[0] + ".json"
                    output_path = os.path.join(
                        args.output_folder,
                        os.path.splitext(os.path.basename(tar_path))[0],
                        relative_path,
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(result_json, f, indent=2, ensure_ascii=False)

        except Exception as e:
            errors.append(
                {
                    "file": fname,
                    "stage": "main_processing",
                    "error": f"{type(e).__name__}: {str(e)}",
                }
            )

    if errors:
        os.makedirs(args.error_log_path, exist_ok=True)
        err_log = os.path.join(
            args.error_log_path,
            f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(err_log, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(errors)} errors to {err_log}")
    else:
        print("No errors found!")


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
    parser.add_argument(
        "--error_log_path", default="", help="Optional path to save error log JSON file"
    )

    args = parser.parse_args()
    main(args)
