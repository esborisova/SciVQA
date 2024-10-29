import pandas as pd
import PIL.Image
import os
import time
import google.generativeai as genai
from ..utils.gemini import save_batch, run_gemini

API_key = ""
scigraphqa_imgs_rootdir = "../../data/scigraphqa_images_2052/"
acl_imgs_rootdir = "../../data/aclfig_images/"
output_filename_root = "../../data/gemini_results/compound_charts/"

def main():
    genai.configure(api_key=API_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    df = pd.read_pickle("../../data/scivqa_data_3000_updated_2024-09-01.pkl")

    batch_index = 1

    chart_ids = []
    captions = []
    types = []

    for index, row in df.iterrows():
        if df["source_dataset"][index] == "aclfig":
            image_path = os.path.join(acl_imgs_rootdir, df["image_file"][index])
        else:
            image_path = os.path.join(
                scigraphqa_imgs_rootdir, df["image_file"][index]
                )

        img = PIL.Image.open(image_path)

        prompt = f'Task: You are given an image of a figure extracted from a scholarly paper and its caption. Identify whether this image contains a compound or non-compound figure. Non-compound means that there is only one figure object in an image. Compound means there are two or more figure objects in an image. If a figure is compound, determine the number of subfigures.\nCaption: {df["caption"][index]}\nOutput Format: JSON containing the figure type. Stick to this format and do not generate any additional text.\nExamples: [{{"compound": "True", "subfigures": "6"}}], [{{"compound": "False", "subfigures": "0"}}].'
        result = run_gemini(model, prompt, img)

        chart_ids.append(df["chart_id"][index])
        types.append(result)
        captions.append(df["caption"][index])

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
