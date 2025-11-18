"""Script to scrape PDFs and LaTeX source files from arXiv and ACL Anthology."""

import pandas as pd
import os
import csv
import argparse
from time import sleep
from ..utils.data_preprocessing import download_file


def main():

    df = pd.read_csv(args.data_path)
    log_file = args.log_path
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["paper_id", "pdf_status", "tex_status", "note"])

    for venue in df["venue"].unique():
        os.makedirs(os.path.join(args.pdf_path, venue), exist_ok=True)
        os.makedirs(os.path.join(args.tex_path, venue), exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        venue = row["venue"]
        pid = row["paper_id"]

        save_dir_pdfs = os.path.join(args.pdf_path, venue)
        save_dir_src = os.path.join(args.tex_path, venue)

        pdf_status, pdf_note = "skipped", ""
        src_status, src_note = "skipped", ""

        pdf_file = os.path.join(save_dir_pdfs, f"{pid.replace('/', '_')}.pdf")

        if venue == "arxiv":
            pdf_url = f"https://arxiv.org/pdf/{pid}.pdf"
            src_url = f"https://arxiv.org/e-print/{pid}"
            src_file = os.path.join(save_dir_src, f"{pid.replace('/', '_')}.tar.gz")
            if not os.path.exists(src_file):
                src_status, src_note = download_file(src_url, src_file)
        else:
            pdf_url = f"http://aclanthology.org/{pid}.pdf"
            src_url = None
            src_status, src_note = "n/a", "no source available"
            src_file = None

        if not os.path.exists(pdf_file):
            pdf_status, pdf_note = download_file(pdf_url, pdf_file)
            print(f"Downloading from arXiv: {pdf_url}, {pdf_file}")

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([pid, pdf_status, src_status, f"{pdf_note} | {src_note}"])

        print(f"{pid} → PDF: {pdf_status}, SRC: {src_status},")

        sleep(5)
        print("Taking a 5-seconds break...")
        if i % 300 == 0:
            print("Taking a 2-minute break...")
            sleep(120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--tex_path", required=True)
    args = parser.parse_args()
    main(args)
