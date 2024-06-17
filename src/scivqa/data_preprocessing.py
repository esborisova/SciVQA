from typing import List
from tqdm import tqdm
import pandas as pd


def clean_text_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].str.strip("[]").str.replace("'", "")
    return df


def detect_en_lang(
    dataset: List[str], nlp, batch_size: int = 10, language_score: int = 0.7
) -> bool:
    """
    Checks which docs in the dataset are English using SpaCy language detection.
    Params:
        dataset (List[str]): A set of text docs.
        nlp: spaCy pipeline.
        batch_size (int: Optional): Number of docs to be processed at a time. By default equals to 10.
        language_score (int: Optional): Language score threshold. Default is 0.7.
    Return:
        bool: True if English, False otherwise
    """

    if "language_detector" not in nlp.pipe_names:
        nlp.add_pipe("language_detector")

    languages = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        docs = list(nlp.pipe(batch))

        for doc in docs:
            languages.append(
                doc._.language == "en" and doc._.language_score >= language_score
            )
    return languages
