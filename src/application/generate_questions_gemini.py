import pandas as pd
import PIL.Image
import os
import time
import google.generativeai as genai
from ..utils.gemini import save_batch

API_key = ""
scigraphqa_imgs_rootdir = "../../data/scigraphqa_images_2052/"
acl_imgs_rootdir = "../../data/aclfig_images/"
output_filename_root = "../../data/gemini_results/"


def main():
    genai.configure(api_key=API_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    df = pd.read_pickle("../../data/scivqa_data_3000.pkl")
    prompts = pd.read_csv("../../data/prompts_gemini.csv")

    batch_index = 1

    chart_ids = []
    qa_pair_types = []
    questions = []
    captions = []

    for index, row in df.iterrows():
        for idx, r in prompts.iterrows():
            if df["source_dataset"][index] == "aclfig":
                image_path = os.path.join(acl_imgs_rootdir, df["image_file"][index])
            else:
                image_path = os.path.join(
                    scigraphqa_imgs_rootdir, df["image_file"][index]
                )

            img = PIL.Image.open(image_path)

            prompt = f'Task: {prompts["task"][idx]}\nCaption: {df["caption"][index]}\nConstraints:\n{prompts["constraints"][idx]}\nOutput Format: {prompts["output_format"][idx]}\nExamples: {prompts["examples"][idx]}'
            try:
                response = model.generate_content([prompt, img])
                result = response.text
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                result = f"Error: {error_type}, error message: {error_message}"

                print(f"Error type: {error_type}")
                print(f"Error message: {error_message}")

            chart_ids.append(df["chart_id"][index])
            qa_pair_types.append(prompts["qa_type"][idx])
            questions.append(result)
            captions.append(df["caption"][index])

        if (index + 1) % 100 == 0:
            print(f"Processed {index} images, saving batch {batch_index}...")

            output_filename = os.path.join(
                output_filename_root, f"batch_{batch_index}.pkl"
            )
            save_batch(
                batch_index,
                output_filename,
                chart_ids,
                captions,
                qa_pair_types,
                questions,
            )
            print(f"Saved batch {batch_index}..")

            chart_ids.clear()
            qa_pair_types.clear()
            questions.clear()
            captions.clear()

            batch_index += 1

            print(f"Sleeping for 60 seconds...")
            time.sleep(60)

    if chart_ids:
        print(f"Saving final batch {batch_index}...")

        output_filename = os.path.join(output_filename_root, f"batch_{batch_index}.pkl")
        save_batch(
            batch_index, output_filename, chart_ids, captions, qa_pair_types, questions
        )


if __name__ == "__main__":
    main()
