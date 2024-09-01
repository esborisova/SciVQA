import pandas as pd
import PIL.Image
import os
import time
import google.generativeai as genai
from ..utils.gemini import save_batch

API_key = ""
scigraphqa_imgs_rootdir = "../../data/scigraphqa_images_2052/"
output_filename_root = "../../data/gemini_results/chart_class_scigraphqa/"


def main():
    genai.configure(api_key=API_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    df = pd.read_pickle("../../data/scivqa_data_3000.pkl")
    scigraph_subset = df[df["source_dataset"] == "scigraphqa"]
    scigraph_subset = scigraph_subset.reset_index(drop=True)

    batch_index = 1

    chart_ids = []
    captions = []
    types = []

    for index, row in scigraph_subset.iterrows():
        image_path = os.path.join(
            scigraphqa_imgs_rootdir, scigraph_subset["image_file"][index]
        )
        img = PIL.Image.open(image_path)

        prompt = f'Task: You are given an image of a figure and its caption extracted from a scholarly paper. Classify this figure into one of the following types: bar chart, box plot, confusion matrix, line chart, map, pie chart, scatter plot, pareto chart, venn diagram, architecture diagram, neural networks, trees.\nCaption: {df["caption"][index]}\nOutput format: JSON, with a single object containing the figure type.\nExample: [{{"type": ""}}].'
        try:
            response = model.generate_content([prompt, img])
            result = response.text
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            result = f"Error: {error_type}, error message: {error_message}"

            print(f"Error type: {error_type}")
            print(f"Error message: {error_message}")

        chart_ids.append(scigraph_subset["chart_id"][index])
        types.append(result)
        captions.append(scigraph_subset["caption"][index])

        if (index + 1) % 15 == 0:
            time.sleep(120)
            print(f"Sleeping for 120 seconds...")

        if (index + 1) % 500 == 0:
            print(f"Processed {index} images, saving batch {batch_index}...")

            output_filename = os.path.join(
                output_filename_root, f"batch_{batch_index}.pkl"
            )
            save_batch(batch_index, output_filename, chart_ids, captions, types)
            chart_ids.clear()
            types.clear()
            captions.clear()

            batch_index += 1

    if chart_ids:
        print(f"Saving final batch {batch_index}...")
        output_filename = os.path.join(output_filename_root, f"batch_{batch_index}.pkl")
        save_batch(batch_index, output_filename, chart_ids, captions, types)


if __name__ == "__main__":
    main()
