""""Prepare metadata for figures and tables."""

import pandas as pd
import os
import argparse
import uuid
from ..utils.data_preprocessing import extract_paper_id, match_figures, extract_meta


def main(args):
    metadata_folder = args.metadata_path
    df = pd.read_pickle(args.data_path)

    # create metadata df
    records = extract_meta(metadata_folder)
    df_meta = pd.DataFrame(records)
    df_meta = df_meta.drop_duplicates(subset=["file_name"], keep="first")
    print(f"Loaded {len(df_meta)} metadata rows")

    # generate unique ids
    df_meta["instance_id"] = df_meta.apply(lambda _: uuid.uuid4().hex, axis=1)

    df_meta = df_meta.rename(columns={"file_name": "image_file"})
    df_meta["figure_id"] = df_meta["image_file"].str.split(".png").str[0]
    df_meta["paper_id"] = df_meta["source_json"].apply(extract_paper_id)
    df_meta = df_meta.drop("source_json", axis=1)
    df_meta.to_pickle(args.save_path)

    # remove duplicates (relevant for scivqa instances)
    images = list(set(df["image_file"].tolist()))
    captions = list(set(df["caption"].tolist()))
    matched_instances = df_meta.apply(match_figures, axis=1, args=(images, captions))

    df_meta_filtered = df_meta[matched_instances == "keep"].reset_index(drop=True)
    df_meta_dropped = df_meta[matched_instances == "drop"].reset_index(drop=True)
    print(f"Kept {len(df_meta_filtered)} rows")
    print(f"Dropped {len(df_meta_dropped)} rows")

    columns_to_add = ["license", "categories", "paper_id"]
    df_unique = df[columns_to_add].drop_duplicates(subset="paper_id")

    df_meta_filtered = df_meta_filtered[
        df_meta_filtered["paper_id"].isin(df_unique["paper_id"])
    ]
    df_meta_filtered = df_meta_filtered.reset_index(drop=True)

    df_merged = pd.merge(df_meta_filtered, df_unique, on="paper_id", how="left")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    df_merged.to_csv(args.save_path, index=False)
    print(f"Saved filtered metadata to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    main(args)
