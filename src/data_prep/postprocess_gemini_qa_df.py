import pandas as pd
import os
from add_figure_types import str_to_json


def main():
    rootdir = "../../data/gemini_results/batches/"
    df = pd.read_pickle("../../data/gemini_results/batches/batch_1.pkl")

    for file in os.listdir(rootdir):
        if file != "batch_1.pkl" and not file.startswith("."):
            qa_df = pd.read_pickle(os.path.join(rootdir, file))
            df = pd.concat([df, qa_df])

    df = df.reset_index(drop=True)

    subset = pd.read_pickle("../../data/gemini_results/regenerated_images.pkl")
    merged_df = pd.merge(
        df,
        subset,
        on=["figure_id", "caption", "qa_pair_type"],
        how="left",
        suffixes=("", "_new"),
    )
    df["gemini_qa_pairs"] = merged_df["gemini_qa_pairs_new"].combine_first(
        df["gemini_qa_pairs"]
    )

    df["gemini_qa_pairs"] = df["gemini_qa_pairs"].apply(
        lambda x: str_to_json(x) if "```json" in x else x
    )
    df.to_pickle("../../data/gemini_results/gemini_qa_pairs.pkl")


if __name__ == "__main__":
    main()
