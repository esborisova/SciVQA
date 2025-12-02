"""Convert output images from MinerU from jpg to png."""

import os
from PIL import Image
import argparse


def main(args):
    root_dir = args.root_dir
    save_dir = args.save_dir
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)

        if not os.path.isfile(file_path):
            continue

        if not file.lower().endswith(".jpg"):
            continue

        im = Image.open(file_path)
        new_file_name = file.split(".jpg")[0] + ".png"
        new_file_path = os.path.join(save_dir, new_file_name)
        im.save(new_file_path)
        print(f"Converted {file} -> {new_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()
    main(args)
