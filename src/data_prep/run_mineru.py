"""Parse PDFs with MinerU (https://github.com/opendatalab/MinerU)"""

import os
import subprocess
import csv
import argparse
import traceback


def main(args):

    pdf_dir = args.input_dir
    out_dir = args.output_dir
    device = args.device
    log_file = args.log_path

    os.makedirs(out_dir, exist_ok=True)

    write_header = not os.path.exists(log_file)
    csv_file = open(log_file, "a", newline="")
    csv_writer = csv.writer(csv_file)

    if write_header:
        csv_writer.writerow(["pdf_path", "error_message"])

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)

            paper_id = os.path.splitext(filename)[0]
            paper_out_dir = os.path.join(out_dir, paper_id)
            if os.path.exists(paper_out_dir):
                print(f"Skipping already processed PDF: {pdf_path}")
                continue

            print(f"Processing: {pdf_path}")

            cmd = ["mineru", "-d", device, "-p", pdf_path, "-o", out_dir]

            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print("Error occurred — continuing to next file.")
                error_msg = traceback.format_exc().replace("\n", "\\n")
                csv_writer.writerow([pdf_path, error_msg])
    csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs one by one with mineru")
    parser.add_argument(
        "-i", "--input_dir", required=True, help="Directory containing PDF files"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output directory for mineru results"
    )
    parser.add_argument(
        "-d", "--device", default="cuda:0", help="Device for mineru (default: cuda:0)"
    )
    parser.add_argument("--log_path", required=True)

    args = parser.parse_args()
    main(args)
