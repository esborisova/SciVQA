"""Create a corpus of 3K documents for annotation. 
Since ACL-fig contains only 948 intries, we randomly sample only data from SciGraphQA."""

import pandas as pd
import shutil
import os
from typing import List


def copy_files(
    rootdir: str,
    target_files: List[str],
    subdirs: List[str] = None,
    dest_dir: str = None,
):
    """
    Copy specified files from rootdir (and optionally subdirs) to dest_dir.

    Args:
        rootdir (str): The root directory to search for files.
        target_files (List[str]): A list of file names to copy.
        subdirs (List[str]): A list of subdirectories to search within rootdir.
        dest_dir (str): The destination directory where files should be copied.
    """
    if subdirs:
        for subdir in subdirs:
            current_dir = os.path.join(rootdir, subdir)
            for file_name in target_files:
                source_file = os.path.join(current_dir, file_name)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_dir)
    else:
        for file_name in target_files:
            source_file = os.path.join(rootdir, file_name)
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_dir)


def main():
    df = pd.read_pickle("../../data/scivqa_data.pkl")
    acl_samples = df[df["source_dataset"] == "aclfig"]
    scigraphqa_samples = df[df["source_dataset"] == "scigraphqa"].sample(
        2052, random_state=42
    )
    result_df = pd.concat([acl_samples, scigraphqa_samples])
    result_df.reset_index(drop=True, inplace=True)
    result_df.to_pickle("../../data/scivqa_data_3000.pkl")

    scigraphqa_imgs_rootdir = "data/scigraphqa_images"
    dest_dir = "data/scigraphqa_images_2052"
    data = df[df["source_dataset"] == "scigraphqa"]
    images = data["image_file"].tolist()
    copy_files(rootdir=scigraphqa_imgs_rootdir, target_files=images, dest_dir=dest_dir)


if __name__ == "__main__":
    main()
