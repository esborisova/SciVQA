import pandas as pd
import re
from tqdm import tqdm

def main():
    """
    This scripts iterates over the full SciGraphQA train dataset (in its four parts) and adds field of research
    categories taken from the arXiv dataset. It then filters and saves CSV files of papers that include cs.CL 
    in their categories. The datasets can be obtained here:
    SciGraphQA: https://huggingface.co/datasets/alexshengzhili/SciGraphQA-295K-train
    arXiv: https://www.kaggle.com/datasets/Cornell-University/arxiv 
    """

    arxiv_data = pd.read_json('arXiv-dataset/arxiv-metadata-oai-snapshot.json', lines=True)

    scigraph_data_paths = [
        'SciGraphQA-dataset/train-00000-of-00004-f92b0b8defd0a6e3.parquet', 
        'SciGraphQA-dataset/train-00001-of-00004-41e4f32726a2e648.parquet',
        'SciGraphQA-dataset/train-00002-of-00004-4d14605cedc8730d.parquet',
        'SciGraphQA-dataset/train-00003-of-00004-3bb8835d6b8cb13f.parquet'
    ]

    for idx, scigraph_data in enumerate(scigraph_data_paths):
    
        scigraphqa_data = pd.read_parquet(scigraph_data)
        
        scigraphqa_data_ids = scigraphqa_data['id'].values
        
        categories = []
        cs_CL = []
        
        for image_id in tqdm(scigraphqa_data_ids):
            arxiv_id = re.sub(r'v\d+$', '', image_id.split('-')[0])
            category = arxiv_data[arxiv_data['id'] == arxiv_id]['categories'].values
            categories.append(category)
            cs_CL.append('cs.CL' in category[0])
            
        scigraphqa_data['categories'] = categories
        scigraphqa_data['cs_CL'] = cs_CL

        scigraphqa_data.to_parquet(str(idx) + '_SciGraphQA_with_categories.parquet')

        sciqarphqa_CL = scigraphqa_data[scigraphqa_data['cs_CL'] == True]

        sciqarphqa_CL.to_csv(str(idx) + '_SciGraphQA_CL.csv')


if __name__ == "__main__":
    main()