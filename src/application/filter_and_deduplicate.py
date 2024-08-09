import pandas as pd
import re
from tqdm import tqdm

def main():

    # 1. Iterate over the full SciGraphQA dataset and save CSV files that only include samples with cs.CL in their categories
    # This also gets the upload date of each paper in the SciGraphQA data
    filter_SciGraphQA_to_CL()

    # 2. Deduplicate the SciGraphQA_CL data by removing all instances connected to papers already included in ACL-Fig
    # This also gets metadata for ACL-Fig from the ACL Anthology
    deduplicate_SciGraphQA()

def filter_SciGraphQA_to_CL():
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
        dates = []
        cs_CL = []
        
        for image_id in tqdm(scigraphqa_data_ids):
            arxiv_id = re.sub(r'v\d+$', '', image_id.split('-')[0])
            category = arxiv_data[arxiv_data['id'] == arxiv_id]['categories'].values
            categories.append(category)
            cs_CL.append('cs.CL' in category[0])
            date = arxiv_data[arxiv_data['id'] == arxiv_id]['upload_date'].values
            dates.append(date)
            
        scigraphqa_data['categories'] = categories
        scigraphqa_data['upload_date'] = dates
        scigraphqa_data['cs_CL'] = cs_CL
        
        scigraphqa_data.to_parquet(str(idx) + '_SciGraphQA_with_categories.parquet')
        
        sciqarphqa_CL = scigraphqa_data[scigraphqa_data['cs_CL'] == True]
        
        sciqarphqa_CL.to_csv(str(idx) + '_SciGraphQA_CL.csv')

def get_ACL_Fig_metadata():
    """
    This script adds metadata to papers in ACL-Fig taken from the ACL Anthology Corpus. 
    ACL-Fig (scientific_figures_pilot.csv): https://huggingface.co/datasets/citeseerx/ACL-fig/tree/main
    ACL Anthology Corpus: https://github.com/shauryr/ACL-anthology-corpus 
    """

    acl_fig_data = pd.read_csv('acl_fig_data.csv')
    acl_anthology = pd.read_parquet('acl-publication-info.74k.parquet')

    abstracts = []
    full_texts = []
    pdf_hashs = []
    numcitedbys = []
    urls = []
    publishers = []
    addresses = []
    years = []
    months = []
    booktitles = []
    authors = []
    titles = []
    dois = []
    journals = []
    
    for idx, row in acl_fig_data.iterrows():
        acl_id = row['acl_paper_id']
        abstracts.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['abstract'].values)
        full_texts.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['full_text'].values)
        pdf_hashs.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['pdf_hash'].values)
        numcitedbys.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['numcitedby'].values)
        urls.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['url'].values)
        publishers.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['publisher'].values)
        addresses.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['address'].values)
        years.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['year'].values)
        months.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['month'].values)
        booktitles.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['booktitle'].values)
        authors.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['author'].values)
        titles.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['title'].values)
        dois.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['doi'].values)
        journals.append(acl_anthology[acl_anthology['acl_id'] == acl_id]['journal'].values)

    acl_fig_data['abstract'] = abstracts
    acl_fig_data['paper_title'] = titles
    acl_fig_data['full_text'] = full_texts
    acl_fig_data['publisher'] = publishers
    acl_fig_data['year'] = years
    acl_fig_data['author'] = authors
    acl_fig_data['doi'] = dois
    acl_fig_data['journal'] = journals

    return acl_fig_data

def deduplicate_SciGraphQA():

    SciGraphQA_CL_0 = pd.read_csv('0_SciGraphQA_CL.csv')
    SciGraphQA_CL_1 = pd.read_csv('1_SciGraphQA_CL.csv')
    SciGraphQA_CL_2 = pd.read_csv('2_SciGraphQA_CL.csv')
    SciGraphQA_CL_3 = pd.read_csv('3_SciGraphQA_CL.csv')

    SciGraphQA_CL = pd.concat([SciGraphQA_CL_0, SciGraphQA_CL_1, SciGraphQA_CL_2, SciGraphQA_CL_3])

    SciGraphQA_CL_titles = SciGraphQA_CL['title'].values
    SciGraphQA_CL_abstracts = SciGraphQA_CL['abstract'].values

    acl_fig_data = get_ACL_Fig_metadata()

    # Iterate over ACL-Fig data and get indices of sampled from SciGraphQA that belong to the same paper
    # Based on exact matches of titles (lowered) OR exact matches of the first 100 chars of abstracts (lowered)

    SciGraphQA_title_dups = []
    SciGraphQA_abstract_dups = []
    
    for idx, row in acl_fig_data.iterrows():
        
        title_row = row['paper_title']
        if len(title_row) != 0:
            title = row['paper_title'][0].lower()
            SciGraphQA_title_indices = [i for i in range(len(SciGraphQA_CL_titles)) if SciGraphQA_CL_titles[i].lower() == title]
            SciGraphQA_title_dups.append(SciGraphQA_title_indices)
        else:
            SciGraphQA_title_dups.append([])
            
        abstract_row = row['abstract']
        if len(abstract_row) != 0:
            abstract = row['abstract'][0][:100].lower()
            SciGraphQA_abstract_indices = [i for i in range(len(SciGraphQA_CL_abstracts)) if SciGraphQA_CL_abstracts[i].replace('\n', ' ')[:100].lower() == abstract]
            SciGraphQA_abstract_dups.append(SciGraphQA_abstract_indices)
            
        else:
            SciGraphQA_abstract_dups.append([])

    acl_fig_data['SciGraphQA_CL_dup_index_titles'] = SciGraphQA_title_dups
    acl_fig_data['SciGraphQA_CL_dup_index_abstracts'] = SciGraphQA_abstract_dups

    # Get all SciGraphQA indices with papers that exist in ACL-Fig and drop them
    overall_scigraph_duplicated_instances = []
    
    for item in SciGraphQA_title_dups:
        if len(item) > 0:
            for idx in item:
                overall_scigraph_duplicated_instances.append(idx)
                
    for item in SciGraphQA_abstract_dups:
        if len(item) > 0:
            for idx in item:
                overall_scigraph_duplicated_instances.append(idx)

    overall_scigraph_duplicated_instances = list(set(overall_scigraph_duplicated_instances))
    SciGraphQA_CL.drop(overall_scigraph_duplicated_instances, inplace=True)
    SciGraphQA_CL.to_csv('SciGraphQA_CL_deduplicated.csv')

    acl_fig_data.to_csv('ACL-Fig_with_metadata.csv')

if __name__ == "__main__":
    main()