import pandas as pd

def create_pdf_url(venue, paper_id):
    """
    Create a PDF URL based on the venue and paper ID.

    Args:
        venue (str): The venue of the paper ('arxiv' or 'acl').
        paper_id (str): The ID of the paper.

    Returns:
        str: The URL of the PDF for the given paper. Returns "Unknown venue" if the venue is not recognized.
    """
    if venue == 'arxiv':
        return f"https://arxiv.org/pdf/{paper_id}"
    elif venue == 'acl':
        return f"https://aclanthology.org/{paper_id}.pdf"
    else:
        return "Unknown venue"

def update_pdf_urls():
    """
    Update the DataFrame with PDF URLs for each paper and save the results.

    This function loads a DataFrame from a pickle file, adds a 'pdf_url' column
    based on the venue and paper_id, prints some information about the data,
    and saves the updated DataFrame back to the same file.
    """
    
    file_path = '../../data/scivqa_data_3000_updated_2024-09-01.pkl'
    df = pd.read_pickle(file_path)
    df['pdf_url'] = df.apply(lambda row: create_pdf_url(row['venue'], row['paper_id']), axis=1)
    
    # print info 
    print(df.columns)
    print(df[['image_file', 'chart_id', 'venue', 'paper_id', 'pdf_url']].head())
    print("All occurring venue types:")
    for venue in df['venue'].unique():
        print(f"- {venue}")
    
    df.to_pickle(file_path)
    print(f"Updated pkl has been saved to {file_path}.")

if __name__ == "__main__":
    update_pdf_urls()