import pandas as pd
import os
from typing import List
from ..utils.data_preprocessing import remove_files


def main():
    source_datasets = ["scigraphqa", "aclfig"]
    aclfig_rootdir = "ACL-fig/training_data/"
    aclfig_dir_meta = "ACL-fig/metadata/"
    aclfig_subdirs = ["train", "test", "val"]
    scigraphqa_rootdir = "SciGraphQA-295K-train/imgs/"

    df = pd.read_pickle("../../data/scivqa_data.pkl")

    for dataset in source_datasets:
        data = df[df["source_dataset"] == dataset]
        images = data["image_file"].tolist()
        if dataset == "aclfig":
            metadata_files = set(data["metadata"].tolist())
            remove_files(
                rootdir=aclfig_rootdir,
                target_files=images,
                subdirs=aclfig_subdirs,
                is_recursive=True,
            )
            remove_files(rootdir=aclfig_dir_meta, target_files=metadata_files)
        remove_files(rootdir=scigraphqa_rootdir, target_files=images, is_recursive=True)


if __name__ == "__main__":
    main()
