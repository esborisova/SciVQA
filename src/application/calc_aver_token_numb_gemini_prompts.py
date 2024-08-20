import pandas as pd
from vertexai.preview import tokenization


def main():
    df = pd.read_pickle("../../data/scivqa_data_3000.pkl")
    prompts = pd.read_csv("../../data/prompts_gemini.csv")

    model_name = "gemini-1.5-flash-001"
    tokenizer = tokenization.get_tokenizer_for_model(model_name)

    all_tokens = []

    for index, row in prompts.iterrows():
        for idx, r in df.iterrows():
            prompt = f'Task: {prompts["task"][index]}\nCaption: {df["caption"][idx]}\nConstraints:\n{prompts["constraints"][index]}\nOutput Format: {prompts["output_format"][index]}\nExamples: {prompts["examples"][index]}'
            n_tokens = tokenizer.count_tokens(prompt)
            all_tokens.append(n_tokens.total_tokens)

    print(
        "Average token number across the prompts is:",
        sum(all_tokens) / (len(all_tokens)),
    )


if __name__ == "__main__":
    main()
