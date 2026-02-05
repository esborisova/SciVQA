"""Collect images not extracted by PDFFigure but available in MinerU outputs."""

import pandas as pd
import os
import json
import argparse
from rapidfuzz import fuzz, process
import re
from ..utils.data_preprocessing import extract_title_prefix


def main(args):

    df = pd.read_pickle(args.data_path)

    paper_ids = set(df["paper_id"].tolist())

    rootdir_mineru = args.mineru_rootdir

    caption_keys = {
        "image": "image_caption",
        "table": "table_caption",
        "figure": "figure_caption",
    }

    footnote_keys = {
        "image": "image_footnote",
        "table": "table_footnote",
        "figure": "figure_footnote",
    }

    new_images = []
    threshold = float(args.sim_threshold)

    for root, _, files in os.walk(rootdir_mineru):
        for file in files:
            # access file with mineru metadata per paper
            if file.endswith(".json") and "content" in file:
                result = file.split("_")[0]
                pid_match = [pid for pid in paper_ids if pid == result]

                if not pid_match:
                    continue

                paper_id = pid_match[0]
                full_path = os.path.join(root, file)

                with open(full_path, "r") as f:
                    data = json.load(f)

                # consider only images with captions/footnotes and image path
                valid_items = [
                    item
                    for item in data
                    if item.get("type") in caption_keys
                    and item.get("img_path")
                    and item["img_path"].strip() != ""
                    and (
                        (
                            item.get(caption_keys[item.get("type")])
                            and len(item[caption_keys[item.get("type")]]) > 0
                        )
                        or (
                            item.get(footnote_keys[item.get("type")])
                            and len(item[footnote_keys[item.get("type")]]) > 0
                        )
                    )
                ]

                mineru_image_counts = len(valid_items)

                df_captions = df[df["paper_id"] == paper_id]["caption"].tolist()
                df_paper_imgs = set(
                    df[df["paper_id"] == paper_id]["image_file"].tolist()
                )
                df_prefixes = {
                    extract_title_prefix(c)
                    for c in df_captions
                    if extract_title_prefix(c)
                }

                if mineru_image_counts == len(df_paper_imgs):
                    continue

                for item in valid_items:
                    caption_list = item.get(caption_keys[item.get("type")], [])
                    footnote_list = item.get(footnote_keys[item.get("type")], [])

                    caption_text = caption_list[0].strip() if caption_list else ""
                    footnote_text = footnote_list[0].strip() if footnote_list else ""

                    prefix_caption = extract_title_prefix(caption_text)
                    prefix_footnote = extract_title_prefix(footnote_text)

                    is_new = True

                    if prefix_caption and prefix_caption in df_prefixes:
                        is_new = False
                    elif prefix_footnote and prefix_footnote in df_prefixes:
                        is_new = False

                    if is_new:
                        # check match based on caption first
                        if caption_text:
                            matches = process.extract(
                                caption_text,
                                df_captions,
                                scorer=fuzz.token_sort_ratio,
                                limit=1,
                            )
                            if not matches or matches[0][1] >= threshold:
                                is_new = False

                        # If no caption or no match, compare based on footnote
                        if is_new and footnote_text:
                            matches_fn = process.extract(
                                footnote_text,
                                df_captions,
                                scorer=fuzz.token_sort_ratio,
                                limit=1,
                            )
                            if not matches_fn or matches_fn[0][1] >= threshold:
                                is_new = False

                    relative_path = item["img_path"]
                    full_img_path = os.path.join(
                        os.path.dirname(full_path), relative_path
                    )

                    if is_new:
                        new_images.append(
                            {
                                "paper_id": paper_id,
                                "img_path": full_img_path,
                                "caption": caption_text,
                                "footnote": footnote_text,
                                "type": item.get("type"),
                            }
                        )

    new_images_df = pd.DataFrame(new_images)
    save_path = args.save_path
    new_images_df.to_pickle(save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the pkl file with already collected images",
    )
    parser.add_argument(
        "--mineru_rootdir", required=True, help="Root directory with mineru results"
    )
    parser.add_argument("--save_path", help="Output path to save results into pkl")
    parser.add_argument("--sim_threshold", help="Threshold for fuzzy match")

    args = parser.parse_args()
    main(args)
