import pandas as pd
import regex as re
import spacy
import spacy_fastlang
from ..scivqa.data_preprocessing import clean_text_columns, detect_en_lang

nlp = spacy.load("en_core_web_lg")

scigraph = pd.read_csv("../data/SciGraphQA_CL_deduplicated_with_dates.csv")
aclfig = pd.read_csv("../data/ACL-Fig_with_metadata.csv")

exclude_chart_types = [
    "tables",
    "algorithms",
    "Screenshots",
    "natural images",
    "word cloud",
    "NLP text_grammar_eg",
]


def main():
    scigraph = pd.read_csv("../../data/SciGraphQA_CL_deduplicated_with_dates.csv")
    aclfig = pd.read_csv("../../data/ACL-Fig_with_metadata.csv")

    scigraph = scigraph.drop(
        columns={"Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0"}
    )
    aclfig = aclfig.drop(columns={"Unnamed: 0.1", "Unnamed: 0"})

    aclfig = aclfig[~aclfig["label"].isin(exclude_chart_types)].reset_index(drop=True)
    aclfig["chart_id"] = [
        re.sub(r"v\d+$", "", fig.split(".png")[0]) for fig in aclfig["sci_fig"]
    ]
    aclfig["venue"] = "acl"
    aclfig["source_dataset"] = "aclfig"
    aclfig.rename(
        columns={
            "sci_fig": "image_file",
            "paper_title": "title",
            "acl_paper_id": "paper_id",
        },
        inplace=True,
    )
    aclfig = clean_text_columns(aclfig, ["year", "abstract", "title", "full_text"])

    scigraph["paper_id"] = [
        re.sub(r"v\d+$", "", image_id.split("-")[0]) for image_id in scigraph["id"]
    ]
    scigraph["venue"] = "arxiv"
    scigraph["source_dataset"] = "scigraphqa"
    scigraph.rename(columns={"update_date": "year", "id": "chart_id"}, inplace=True)
    scigraph = scigraph.drop(columns={"updated_cs_CL", "cs_CL"})
    scigraph = clean_text_columns(scigraph, ["year"])
    scigraph["year"] = pd.to_datetime(scigraph["year"]).dt.year.astype(str)

    df = pd.concat([scigraph, aclfig], ignore_index=True)
    empty_values = df[
        (df["year"].fillna("") == "")
        | (df["abstract"].fillna("") == "")
        | (df["title"].fillna("") == "")
    ].index
    df = df.drop(empty_values).reset_index(drop=True)

    lang_titles = detect_en_lang(df["title"].tolist(), nlp, batch_size=50)
    df["title_lang"] = lang_titles

    lang_abstracts = detect_en_lang(df["abstract"].tolist(), nlp, batch_size=50)
    df["abstract_lang"] = lang_abstracts

    df = df[df["abstract_lang"] == True]
    df = df.drop(columns={"title_lang", "abstract_lang"}).reset_index(drop=True)

    df.to_pickle("../../data/scivqa_data.pkl")


if __name__ == "__main__":
    main()
