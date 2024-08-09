from typing import List
from tqdm import tqdm
import pandas as pd
import os
import shutil


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


def remove_files(
    rootdir: str,
    target_files: List[str],
    subdirs: List[str] = None,
    is_recursive: bool = False,
):
    """
    Removes files not in the target list from the specified directory (and optionally its subdirectories).

    Args:
        rootdir (str): The root directory to start searching for files.
        target_files (List[str]): List of filenames to keep.
        subdirs (List[str], optional): List of subdirectories within the rootdir to search (if not specified, searches rootdir directly).
        is_recursive (bool, optional): Whether to search recursively in subdirectories.
    """

    def remove_in_directory(directory):
        for item in os.listdir(directory):
            if not item.startswith("."):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path) and is_recursive:
                    remove_in_directory(item_path)
                elif os.path.isfile(item_path) and item not in target_files:
                    os.remove(item_path)

    if subdirs:
        for subdir in subdirs:
            path = os.path.join(rootdir, subdir)
            if os.path.isdir(path):
                remove_in_directory(path)
    else:
        remove_in_directory(rootdir)


def copy_files(
    rootdir: str,
    target_files: List[str],
    subdirs: List[str] = None,
    dest_dir: str = None,
):
    """
    Copy specified files from rootdir (and optionally subdirs) to dest_dir.

    Args:
        rootdir (str): The root directory to search for files.
        target_files (List[str]): A list of file names to copy.
        subdirs (List[str]): A list of subdirectories to search within rootdir.
        dest_dir (str): The destination directory where files should be copied.
    """
    if subdirs:
        for subdir in subdirs:
            current_dir = os.path.join(rootdir, subdir)
            for file_name in target_files:
                source_file = os.path.join(current_dir, file_name)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_dir)
    else:
        for file_name in target_files:
            source_file = os.path.join(rootdir, file_name)
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_dir)
