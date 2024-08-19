"""Create a corpus of 3K documents for annotation. 
Since ACL-fig contains only 948 intries, we randomly sample only data from SciGraphQA"""

import pandas as pd
from ..scivqa.data_preprocessing import copy_files


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
