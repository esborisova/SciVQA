import pandas as pd
import PIL.Image
import os
import time
import google.generativeai as genai

API_key = ""
scigraphqa_imgs_rootdir = "../../data/scigraphqa_images_2052/"
acl_imgs_rootdir = "../../data/aclfig_images/"


def main():
    genai.configure(api_key=API_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    df = pd.read_pickle("../../data/scivqa_data_3000.pkl")
    prompts = pd.read_csv("../../data/prompts_gemini.csv")

    batch_index = 1
    count = 0
    
    chart_ids = []
    qa_pair_types = []
    questions = []


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
            response = model.generate_content([prompt, img])
            result = response.text

            chart_ids.append(df["chart_id"][index])
            qa_pair_types.append(prompts["qa_type"][idx])
            questions.append(result)
        
        count += 1

        if count % 500 == 0:
            print(f"Processed {count} images, saving batch {batch_index}...")
            
            columns = ["chart_id", "qa_pair_type", "gemini_generated_questions"]
            results_df = pd.DataFrame(columns=columns)

            results_df["chart_id"] = chart_ids
            results_df["qa_pair_type"] = qa_pair_types
            results_df["gemini_generated_questions"] = questions

            output_filename = f"../../data/gemini/batch_{batch_index}.pkl"
            results_df.to_pickle(output_filename)
            print(f"Saved batch {batch_index} to {output_filename}")

            chart_ids.clear()
            qa_pair_types.clear()
            questions.clear()

            batch_index += 1
            
            print(f"Sleeping for 60 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    main()
