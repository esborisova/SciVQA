import pandas as pd
import os
import datetime

from ..utils.post_processing import str_to_json, convert_to_list


def main():
    rootdir = "../../data/gemini_results/chart_class_scigraphqa/"
    df = pd.read_pickle("../../data/gemini_results/chart_class_scigraphqa/batch_1.pkl")
    scivqa_data = pd.read_pickle("../../data/scivqa_data_3000.pkl")
    scivqa_data = scivqa_data.rename(columns={"label": "chart_type"})

    for file in os.listdir(rootdir):
        if file != "batch_1.pkl" and not file.startswith("."):
            types_df = pd.read_pickle(os.path.join(rootdir, file))
            df = pd.concat([df, types_df])

    df = df.reset_index(drop=True)

    df["chart_type"] = df["chart_type"].apply(
        lambda x: str_to_json(x) if "```json" in x else x
    )

    df["chart_type"] = df["chart_type"].apply(convert_to_list)

    df.to_pickle("../../data/gemini_results/chart_types_scigraphqa_subset.pkl")

    scigraphqa_types = df["chart_type"].tolist()
    chart_types = [item[0]["type"] for item in scigraphqa_types]
    filtered_df = scivqa_data[scivqa_data["source_dataset"] == "scigraphqa"].copy()
    filtered_df["chart_type"] = chart_types
    scivqa_data.update(filtered_df)

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    scivqa_data.to_pickle(f"../../data/scivqa_data_3000_updated_{date_str}.pkl")


if __name__ == "__main__":
    main()
