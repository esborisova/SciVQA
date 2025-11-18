"""Sample 3000 arXiv papers."""

import argparse
import pandas as pd
from ..utils.data_preprocessing import combine_id_versions


allowed_licences = [
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by-sa/4.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/3.0/",
]


def main(args):
    arxiv = pd.read_pickle(args.arxiv_bulk)
    scivqa = pd.read_pickle(args.scivqa_data)  # already filtered data by licence

    # filter by lucense
    arxiv = arxiv[arxiv["license"].isin(allowed_licences)]
    arxiv["license"].value_counts()

    arxiv["paper_id"] = arxiv.apply(combine_id_versions, axis=1)

    # filter our papers with several versions
    arxiv = arxiv[arxiv["paper_id"].apply(len) == 1].copy()
    arxiv["paper_id"] = arxiv["paper_id"].apply(
        lambda x: str(x[0]) if isinstance(x, list) and len(x) > 0 else ""
    )
    arxiv = arxiv.reset_index(drop=True)

    # exclude papers already present in scivqa
    scivqa_ids = scivqa["arxiv_id"].tolist()
    arxiv = arxiv[~arxiv["paper_id"].isin(scivqa_ids)]

    sampled_arxiv = arxiv.sample(3000, random_state=42)
    sampled_arxiv = sampled_arxiv.reset_index(drop=True)
    sampled_arxiv.to_pickle(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv_bulk", required=True)
    parser.add_argument("--scivqa_data", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
