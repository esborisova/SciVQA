"""Add images extracted with MinerU to those previously extracted with PDFFigure."""

import pandas as pd
import argparse
from ..utils.data_preprocessing import add_filenames, generate_unique_id


def main(args):

    df = pd.read_pickle(args.data_path)
    new_images_df = pd.read_pickle(args.new_images)

    # add file names following pdffigure convention
    new_images_df["image_file"] = add_filenames(new_images_df)
    new_images_df = new_images_df.dropna(subset=["image_file"])
    new_images_df = new_images_df.reset_index(drop=True)

    mask = new_images_df["caption"].str.strip() == ""
    new_images_df.loc[mask & new_images_df["footnote"].notna(), "caption"] = (
        new_images_df.loc[mask & new_images_df["footnote"].notna(), "footnote"]
    )
    new_images_df = new_images_df.drop(["footnote", "type"], axis=1)

    merged_df = pd.concat(
        [df, new_images_df[["paper_id", "image_file", "caption"]]], ignore_index=True
    )

    # avoid overlap with unique ids for target figures
    existing_ids = set(df["instance_id"])
    merged_df["instance_id"] = merged_df.apply(
        lambda _: generate_unique_id(existing_ids), axis=1
    )

    merged_df["figure_id"] = merged_df["image_file"].str.replace(
        r"\.png$", "", regex=True
    )

    # add metadata to new image instances
    columns_to_copy = ["license", "categories", "venue", "source_dataset", "pdf_url"]
    paper_info = merged_df.groupby("paper_id")[columns_to_copy].first()

    for col in columns_to_copy:
        mask = merged_df[col].isna()
        merged_df.loc[mask, col] = merged_df.loc[mask, "paper_id"].map(paper_info[col])

    merged_df.to_pickle(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the pkl file with already collected images",
    )
    parser.add_argument(
        "--new_images",
        required=True,
        help="Path to the pkl file with images collected with MinerU",
    )
    parser.add_argument("--output_path", help="Output path to save results into pkl")
    args = parser.parse_args()
    main(args)
