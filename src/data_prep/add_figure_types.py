"Add type labels generated with Gemini for SciGraphQA instances."

import pandas as pd
import os
import datetime
import json
import ast


def str_to_json(text: str):
    try:
        cleaned_str = text.replace("```json\n", "").replace("\n```", "")
        json_data = json.loads(cleaned_str)
        return json_data

    except Exception as e:
        return text


def convert_to_list(val: str):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val


def main():
    rootdir = "../../data/gemini_results/fig_class_scigraphqa/"
    df = pd.read_pickle("../../data/gemini_results/fig_class_scigraphqa/batch_1.pkl")
    scivqa_data = pd.read_pickle("../../data/scivqa_data_3000.pkl")
    scivqa_data = scivqa_data.rename(columns={"label": "figure_type"})

    for file in os.listdir(rootdir):
        if file != "batch_1.pkl" and not file.startswith("."):
            types_df = pd.read_pickle(os.path.join(rootdir, file))
            df = pd.concat([df, types_df])

    df = df.reset_index(drop=True)

    df["figure_type"] = df["figure_type"].apply(
        lambda x: str_to_json(x) if "```json" in x else x
    )

    df["figure_type"] = df["figure_type"].apply(convert_to_list)

    df.to_pickle("../../data/gemini_results/fig_types_scigraphqa_subset.pkl")

    scigraphqa_types = df["figure_type"].tolist()
    fig_types = [item[0]["type"] for item in scigraphqa_types]
    filtered_df = scivqa_data[scivqa_data["source_dataset"] == "scigraphqa"].copy()
    filtered_df["figure_type"] = fig_types
    scivqa_data.update(filtered_df)

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    scivqa_data.to_pickle(f"../../data/scivqa_data_3000_updated_{date_str}.pkl")


if __name__ == "__main__":
    main()
