import pandas as pd


def create_df(chart_ids, captions, types, questions=None):
    if questions:
        columns = ["chart_id", "caption", "qa_pair_type", "gemini_qa_pairs"]

        results_df = pd.DataFrame(columns=columns)
        results_df["chart_id"] = chart_ids
        results_df["caption"] = captions
        results_df["qa_pair_type"] = types
        results_df["gemini_qa_pairs"] = questions

    columns = ["chart_id", "caption", "chart_type"]

    results_df = pd.DataFrame(columns=columns)
    results_df["chart_id"] = chart_ids
    results_df["caption"] = captions
    results_df["chart_type"] = types

    return results_df


def save_batch(
    batch_index, output_filename, chart_ids, captions, types, questions=None
):
    if questions:
        df = create_df(chart_ids, captions, types, questions)
    df = create_df(chart_ids, captions, types)

    df.to_pickle(output_filename)
    print(f"Saved batch {batch_index} to {output_filename}")
