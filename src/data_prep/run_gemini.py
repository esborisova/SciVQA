"""Script for running Gemini: 
1. type classification for SciGraphQA instances,
2. classifying figures into compound and non-compound,
3. QA pairs generation."""

import argparse
import os
import time
import pandas as pd
import PIL.Image
import google.generativeai as genai


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
    fig_ids: list, captions: list, types: list, questions: list = None
) -> pd.DataFrame:
    if questions is not None:
        results_df = pd.DataFrame(
            {
                "figure_id": fig_ids,
                "caption": captions,
                "qa_pair_type": types,
                "gemini_qa_pairs": questions,
            }
        )

    else:
        results_df = pd.DataFrame(
            {"figure_id": fih_ids, "caption": captions, "figure_type": types}
        )
    return results_df


def save_batch(
    batch_index: int,
    output_filename: str,
    fig_ids: list,
    captions: list,
    types: list,
    questions: list = None,
):
    if questions is not None:
        df = create_df(fig_ids, captions, types, questions)
    else:
        df = create_df(fig_ids, captions, types)

    df.to_pickle(output_filename)
    print(f"Saved batch {batch_index} to {output_filename}")


def configure_model(api_key, model_name="gemini-1.5-flash"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def get_image_path(row, acl_dir, scigraph_dir):
    if row["source_dataset"] == "aclfig":
        return os.path.join(acl_dir, row["image_file"])
    return os.path.join(scigraph_dir, row["image_file"])


def put_to_sleep(duration=120):
    print(f"Sleeping for {duration} seconds...")
    time.sleep(duration)


def build_qa_prompt(task_desc, caption, constraints, output_format, examples):
    return (
        f"Task: {task_desc}\n"
        f"Caption: {caption}\n"
        f"Constraints:\n{constraints}\n"
        f"Output Format: {output_format}\n"
        f"Examples: {examples}"
    )


def save_and_clear_batch(
    batch_index,
    output_dir,
    figure_ids,
    captions,
    qa_pair_types=None,
    questions=None,
    types=None,
):
    output_filename = os.path.join(output_dir, f"batch_{batch_index}.pkl")
    save_batch(
        batch_index,
        output_filename,
        figure_ids,
        captions,
        qa_pair_types or types,
        questions,
    )
    figure_ids.clear()
    captions.clear()
    if qa_pair_types is not None:
        qa_pair_types.clear()
    if questions is not None:
        questions.clear()
    if types is not None:
        types.clear()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model = configure_model(args.api_key, args.model_name)
    df = pd.read_pickle(args.data_pickle)

    batch_index = 1
    figure_ids, captions, qa_pair_types, questions, types = [], [], [], [], []

    if args.task == "qa_generation":
        if not args.prompts_file:
            raise ValueError("--prompts_file is required for qa_generation task.")
        if not args.acl_dir:
            raise ValueError("--acl_dir is required for qa_generation task.")

        prompts = pd.read_csv(args.prompts_file)
        df = df[:2]
        for index, row in df.iterrows():
            for _, prompt_row in prompts.iterrows():
                try:
                    image_path = get_image_path(row, args.acl_dir, args.scigraph_dir)
                    img = PIL.Image.open(image_path)
                    prompt = build_qa_prompt(
                        prompt_row["task"],
                        row["caption"],
                        prompt_row["constraints"],
                        prompt_row["output_format"],
                        prompt_row["examples"],
                    )

                    result = run_gemini(model, prompt, img)

                    figure_ids.append(row["figure_id"])
                    qa_pair_types.append(prompt_row["qa_type"])
                    questions.append(result)
                    captions.append(row["caption"])

                    if len(figure_ids) >= 100:
                        print(
                            f"Processed {index} images, saving batch {batch_index}..."
                        )
                        save_and_clear_batch(
                            batch_index,
                            args.output_dir,
                            figure_ids,
                            captions,
                            qa_pair_types,
                            questions,
                        )
                        batch_index += 1
                        put_to_sleep()
                except Exception as e:
                    print(f"Error processing index {index}: {e}")
        if figure_ids:
            print(f"Saving final batch {batch_index}...")
            save_and_clear_batch(
                batch_index,
                args.output_dir,
                figure_ids,
                captions,
                qa_pair_types,
                questions,
            )

    elif args.task == "compound_classification":
        if not args.acl_dir:
            raise ValueError("--acl_dir is required for this task.")
        for index, row in df.iterrows():
            try:
                image_path = get_image_path(row, args.acl_dir, args.scigraph_dir)
                img = PIL.Image.open(image_path)
                prompt = (
                    "Task: You are given an image of a figure extracted from a scholarly paper and its caption. "
                    "Identify whether this image contains a compound or non-compound figure. Non-compound means that there is only one "
                    "figure object in an image. Compound means there are two or more figure objects in an image. If a figure is compound, "
                    "determine the number of subfigures.\n"
                    f"Caption: {row['caption']}\n"
                    "Output Format: JSON containing the figure type.\n"
                    'Examples: [{"compound": "True", "subfigures": "6"}], [{"compound": "False", "subfigures": "0"}]'
                )
                result = run_gemini(model, prompt, img)
                figure_ids.append(row["figure_id"])
                types.append(result)
                captions.append(row["caption"])

                if len(figure_ids) >= 100:
                    print(f"Processed {index} images, saving batch {batch_index}...")
                    save_and_clear_batch(
                        batch_index, args.output_dir, figure_ids, captions, types=types
                    )
                    batch_index += 1

                    put_to_sleep()
            except Exception as e:
                print(f"Error processing index {index}: {e}")

        if figure_ids:
            print(f"Saving final batch {batch_index}...")
            save_and_clear_batch(
                batch_index, args.output_dir, figure_ids, captions, types=types
            )

    elif args.task == "type_classification":
        if not args.acl_dir:
            raise ValueError("--acl_dir is required for this task.")
        scigraph_subset = df[df["source_dataset"] == "scigraphqa"]
        scigraph_subset = scigraph_subset.reset_index(drop=True)
        for index, row in scigraph_subset.iterrows():
            try:
                image_path = os.path.join(args.scigraph_dir, row["image_file"])
                img = PIL.Image.open(image_path)
                prompt = (
                    "Task: You are given an image of a figure and its caption extracted from a scholarly paper."
                    "Classify this figure into one of the following types: bar chart, box plot, confusion matrix, "
                    "line chart, pie chart, scatter plot, pareto chart, venn diagram, architecture diagram, neural networks, trees."
                    f"Caption: {row['caption']}\n"
                    "Output format: JSON, with a single object containing the figure type.\n"
                    'Example: [{"type": ""}].'
                )
                result = run_gemini(model, prompt, img)
                figure_ids.append(row["figure_id"])
                types.append(result)
                captions.append(row["caption"])

                if len(figure_ids) >= 100:
                    print(f"Processed {index} images, saving batch {batch_index}...")
                    save_and_clear_batch(
                        batch_index, args.output_dir, figure_ids, captions, types=types
                    )
                    batch_index += 1
                    put_to_sleep()

            except Exception as e:
                print(f"Error processing index {index}: {e}")

        if figure_ids:
            print(f"Saving final batch {batch_index}...")
            save_and_clear_batch(
                batch_index, args.output_dir, figure_ids, captions, types=types
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=["qa_generation", "compound_classification", "type_classification"],
        help="Type of task to run",
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--scigraph_dir", required=True)
    parser.add_argument("--acl_dir", required=False)
    parser.add_argument("--data_pickle", required=True)
    parser.add_argument("--prompts_file", required=False)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    main(args)
