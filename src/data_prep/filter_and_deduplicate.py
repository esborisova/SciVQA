"Filter and deduplicate SciGraphQA and ACL-Fig."

import pandas as pd
import re
from tqdm import tqdm
from typing import List


def add_arxiv_metadata(
    scigraph_df: pd.DataFrame, arxiv_data: pd.DataFrame
) -> pd.DataFrame:
    categories, dates, cs_cl = [], [], []

    for image_id in tqdm(scigraph_df["id"]):
        arxiv_id = re.sub(r"v\d+$", "", image_id.split("-")[0])
        metadata = arxiv_data.loc.get(arxiv_id, {})

        category = metadata.get("categories", [])
        categories.append(category)
        cs_cl.append("cs.CL" in category if category else False)
        dates.append(metadata.get("upload_date", None))

    scigraph_df["categories"] = categories
    scigraph_df["upload_date"] = dates
    scigraph_df["cs_CL"] = cs_cl

    return scigraph_df


def filter_SciGraphQA_to_CL(arxiv_data_path: str, scigraph_data_paths: List[str]):
    """
    Iterates over the full SciGraphQA train dataset (in its four parts) and adds field of research
    categories taken from the arXiv dataset. It then filters and saves CSV files of papers that include cs.CL
    in their categories.
    """
    arxiv_data = pd.read_json(arxiv_data_path, lines=True)
    for idx, scigraph_data in enumerate(scigraph_data_paths):
        scigraphqa_data = pd.read_parquet(scigraph_data)
        scigraphqa_data = add_arxiv_metadata(scigraph_data, arxiv_data)

        scigraphqa_data.to_parquet(f"{idx}_SciGraphQA_with_categories.parquet")
        sciqarphqa_CL = scigraphqa_data[scigraphqa_data["cs_CL"] == True]
        sciqarphqa_CL.to_csv(str(idx) + "_SciGraphQA_CL.csv")
        sciqarphqa_CL[sciqarphqa_CL["cs_CL"]].to_csv(f"{idx}_SciGraphQA_CL.csv")


def get_ACL_Fig_metadata(
    acl_fig_data_path: str, acl_anthology_path: str
) -> pd.DataFrame:
    """
    Adds metadata to papers in ACL-Fig taken from the ACL Anthology Corpus.
    """
    acl_fig_data = pd.read_csv(acl_fig_data_path)
    acl_anthology = pd.read_parquet(acl_anthology_path)

    abstracts = []
    full_texts = []
    pdf_hashs = []
    numcitedbys = []
    urls = []
    publishers = []
    addresses = []
    years = []
    months = []
    booktitles = []
    authors = []
    titles = []
    dois = []
    journals = []

    for _, row in acl_fig_data.iterrows():
        acl_id = row["acl_paper_id"]
        abstracts.append(
            acl_anthology[acl_anthology["acl_id"] == acl_id]["abstract"].values
        )
        full_texts.append(
            acl_anthology[acl_anthology["acl_id"] == acl_id]["full_text"].values
        )
        pdf_hashs.append(
            acl_anthology[acl_anthology["acl_id"] == acl_id]["pdf_hash"].values
        )
        numcitedbys.append(
            acl_anthology[acl_anthology["acl_id"] == acl_id]["numcitedby"].values
        )
        urls.append(acl_anthology[acl_anthology["acl_id"] == acl_id]["url"].values)
        publishers.append(
            acl_anthology[acl_anthology["acl_id"] == acl_id]["publisher"].values
        )
        addresses.append(
            acl_anthology[acl_anthology["acl_id"] == acl_id]["address"].values
        )
        years.append(acl_anthology[acl_anthology["acl_id"] == acl_id]["year"].values)
        months.append(acl_anthology[acl_anthology["acl_id"] == acl_id]["month"].values)
        booktitles.append(
            acl_anthology[acl_anthology["acl_id"] == acl_id]["booktitle"].values
        )
        authors.append(
            acl_anthology[acl_anthology["acl_id"] == acl_id]["author"].values
        )
        titles.append(acl_anthology[acl_anthology["acl_id"] == acl_id]["title"].values)
        dois.append(acl_anthology[acl_anthology["acl_id"] == acl_id]["doi"].values)
        journals.append(
            acl_anthology[acl_anthology["acl_id"] == acl_id]["journal"].values
        )

    acl_fig_data["abstract"] = abstracts
    acl_fig_data["paper_title"] = titles
    acl_fig_data["full_text"] = full_texts
    acl_fig_data["publisher"] = publishers
    acl_fig_data["year"] = years
    acl_fig_data["author"] = authors
    acl_fig_data["doi"] = dois
    acl_fig_data["journal"] = journals

    return acl_fig_data


def load_scigraph_CL_files(files_path: List[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(file) for file in files_path], ignore_index=True)


def deduplicate_SciGraphQA(
    aclfig_data_path: str,
    acl_anthology_path: str,
    output_scigraphqa_path: str,
    output_aclfig_path: str,
):
    scigraph_files = [f"{i}_SciGraphQA_CL.csv" for i in range(4)]
    SciGraphQA_CL = load_scigraph_CL_files(scigraph_files)

    SciGraphQA_CL_titles = SciGraphQA_CL["title"].values
    SciGraphQA_CL_abstracts = SciGraphQA_CL["abstract"].values

    acl_fig_data = get_ACL_Fig_metadata(aclfig_data_path, acl_anthology_path)

    SciGraphQA_title_dups = []
    SciGraphQA_abstract_dups = []
    for idx, row in acl_fig_data.iterrows():
        title_row = row["paper_title"]
        if len(title_row) != 0:
            title = row["paper_title"][0].lower()
            SciGraphQA_title_indices = [
                i
                for i in range(len(SciGraphQA_CL_titles))
                if SciGraphQA_CL_titles[i].lower() == title
            ]
            SciGraphQA_title_dups.append(SciGraphQA_title_indices)
        else:
            SciGraphQA_title_dups.append([])

        abstract_row = row["abstract"]
        if len(abstract_row) != 0:
            abstract = row["abstract"][0][:100].lower()
            SciGraphQA_abstract_indices = [
                i
                for i in range(len(SciGraphQA_CL_abstracts))
                if SciGraphQA_CL_abstracts[i].replace("\n", " ")[:100].lower()
                == abstract
            ]
            SciGraphQA_abstract_dups.append(SciGraphQA_abstract_indices)

        else:
            SciGraphQA_abstract_dups.append([])

    acl_fig_data["SciGraphQA_CL_dup_index_titles"] = SciGraphQA_title_dups
    acl_fig_data["SciGraphQA_CL_dup_index_abstracts"] = SciGraphQA_abstract_dups

    overall_scigraph_duplicated_instances = []

    for item in SciGraphQA_title_dups:
        if len(item) > 0:
            for idx in item:
                overall_scigraph_duplicated_instances.append(idx)

    for item in SciGraphQA_abstract_dups:
        if len(item) > 0:
            for idx in item:
                overall_scigraph_duplicated_instances.append(idx)

    overall_scigraph_duplicated_instances = list(
        set(overall_scigraph_duplicated_instances)
    )

    SciGraphQA_CL.drop(overall_scigraph_duplicated_instances, inplace=True)
    SciGraphQA_CL.to_csv(output_scigraphqa_path, index=False)
    acl_fig_data.to_csv(output_aclfig_path, index=False)


def main():
    scigraph_data_paths = [
        "SciGraphQA-dataset/train-00000-of-00004-f92b0b8defd0a6e3.parquet",
        "SciGraphQA-dataset/train-00001-of-00004-41e4f32726a2e648.parquet",
        "SciGraphQA-dataset/train-00002-of-00004-4d14605cedc8730d.parquet",
        "SciGraphQA-dataset/train-00003-of-00004-3bb8835d6b8cb13f.parquet",
    ]

    filter_SciGraphQA_to_CL(
        arxiv_data_path="arXiv-dataset/arxiv-metadata-oai-snapshot.json",
        scigraph_data_paths=scigraph_data_paths,
    )

    deduplicate_SciGraphQA(
        aclfig_data_path="acl_fig_data.csv",
        acl_anthology_path="acl-publication-info.74k.parquet",
        output_scigraphqa_path="SciGraphQA_CL_deduplicated.csv",
        output_aclfig_path="ACL-Fig_with_metadata.csv",
    )


if __name__ == "__main__":
    main()
