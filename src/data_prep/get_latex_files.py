import requests
import tarfile
import shutil
import pandas as pd
from tqdm import tqdm
import pickle
import os
import re
import time

def main():

    # Load current SciVQA full data
    with open('scivqa_data.pkl', 'rb') as f:
        scivqa_data = pickle.load(f)

    # Filter to papers originating from SciGraphQA
    scivqa_data_arxiv = scivqa_data[scivqa_data['source_dataset'] == 'scigraphqa']
    
    scivqa_data_arxiv['full_text_status'] = "Not processed"

    save_directory = "/netscratch/abu/Shared-Tasks/SciVQA/arxiv-full-text/tar-files"
    os.makedirs(save_directory, exist_ok=True)

    with open('tex_status.pkl', 'rb') as f:
        tex_status = pickle.load(f)

    for idx, row in tqdm(scivqa_data_arxiv.iterrows()):
        image_id = row['image_file']
        arxiv_id = re.sub(r'v\d+$', '', image_id.split('-')[0])
        
        # Form the source file URL
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        source_filename = os.path.join(save_directory, f"{arxiv_id}.tar.gz")
        
        try:
            # Download the source file
            response = requests.get(source_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            with open(source_filename, 'wb') as source_file:
                source_file.write(response.content)
            print(f"Downloaded {source_filename}")
            
            # Extract the tar.gz file
            extract_dir = os.path.join(save_directory, arxiv_id)
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(source_filename, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            print(f"Extracted {source_filename} to {extract_dir} at idx {idx}")
            
            # Create a new tar.gz file with only .tex files
            tex_files_tar_path = os.path.join(save_directory, f"{arxiv_id}_tex_files.tar.gz")
            tex_files_found = False
            with tarfile.open(tex_files_tar_path, "w:gz") as tex_tar:
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if file.endswith(".tex"):
                            tex_file_path = os.path.join(root, file)
                            tex_tar.add(tex_file_path, arcname=os.path.relpath(tex_file_path, extract_dir))
                            tex_files_found = True

            if tex_files_found:
                scivqa_data_arxiv.at[idx, 'full_text_status'] = "Full text downloaded"
                print(f"Compressed .tex files to {tex_files_tar_path}")
            else:
                scivqa_data_arxiv.at[idx, 'full_text_status'] = "No .tex files found"
                print(f"No .tex files found for {arxiv_id}")
            
        except requests.HTTPError as e:
            scivqa_data_arxiv.at[idx, 'full_text_status'] = f"Failed to download source: {e}"
            print(f"Failed to download source for {arxiv_id}: {e}")

        except tarfile.ReadError as e:
            scivqa_data_arxiv.at[idx, 'full_text_status'] = f"Failed to extract tar file: {e}"
            print(f"Failed to extract tar file for {arxiv_id}: {e}")

        except Exception as e:
            scivqa_data_arxiv.at[idx, 'full_text_status'] = f"An error occurred: {e}"
            print(f"An error occurred with {arxiv_id}: {e}")

        finally:
            # Clean up: delete the extracted directory and the original tar.gz file
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
                print(f"Deleted directory {extract_dir}")
            if os.path.exists(source_filename):
                os.remove(source_filename)
                print(f"Deleted file {source_filename}")

        time.sleep(10)
        
    scivqa_data_arxiv.to_csv('scivqa_data_arxiv_full_text_status.csv', index=False)
    print("All source files processed and directories cleaned up. Status saved to scivqa_data_arxiv_full_text_status.csv.")


if __name__ == "__main__":
    main()
