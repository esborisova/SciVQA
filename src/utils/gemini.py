import pandas as pd


def run_gemini(model, prompt: str, img) -> str:
    try:
        response = model.generate_content([prompt, img])
        result = response.text
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        result = f"Error: {error_type}, error message: {error_message}"

        print(f"Error type: {error_type}")
        print(f"Error message: {error_message}")
    return result


def create_df(
    chart_ids: list, captions: list, types: list, questions: list = None
) -> pd.DataFrame:
    if questions is not None:
        results_df = pd.DataFrame(
            {
                "chart_id": chart_ids,
                "caption": captions,
                "qa_pair_type": types,
                "gemini_qa_pairs": questions,
            }
        )

    else:
        results_df = pd.DataFrame(
            {"chart_id": chart_ids, "caption": captions, "chart_type": types}
        )
    return results_df


def save_batch(
    batch_index: int,
    output_filename: str,
    chart_ids: list,
    captions: list,
    types: list,
    questions: list = None,
):
    if questions is not None:
        df = create_df(chart_ids, captions, types, questions)
    else:
        df = create_df(chart_ids, captions, types)

    df.to_pickle(output_filename)
    print(f"Saved batch {batch_index} to {output_filename}")
