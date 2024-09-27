import pickle
import json
from typing import Dict, List, Any
import pandas as pd

def load_pickle_data(file_path: str) -> Any:
    """Load data from a pickle file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def convert_qa_pair_type(qa_type: str) -> str:
    """Convert qa_pair_type to a simplified format."""
    if 'visual' in qa_type:
        return 'visual'
    elif 'non-visual' in qa_type:
        return 'non-visual'
    return qa_type

def is_binary(qa_type: str) -> bool:
    """Check if the question is binary."""
    return 'binary' in qa_type

def has_options(qa_pair: Dict[str, Any]) -> bool:
    """Check if the question has options."""
    return 'options' in qa_pair

def process_qa_pairs(chart_df: pd.DataFrame) -> Dict[str, Any]:
    """Process and flatten QA pairs."""
    qa_pairs_flat = {}
    pair_counter = 1
    
    for _, row in chart_df.iterrows():
        qa_pair_type = row['qa_pair_type']
        gemini_qa_pairs = row['gemini_qa_pairs']
        
        for qa_pair in gemini_qa_pairs:
            qa_pairs_flat[f"qa_pair_type_{pair_counter}"] = convert_qa_pair_type(qa_pair_type)
            qa_pairs_flat[f"question_{pair_counter}"] = qa_pair['question']
            qa_pairs_flat[f"answer_{pair_counter}"] = qa_pair.get('answer', '')
            qa_pairs_flat[f"is_binary_{pair_counter}"] = is_binary(qa_pair_type)
            qa_pairs_flat[f"has_options_{pair_counter}"] = has_options(qa_pair)
            qa_pairs_flat[f"is_unanswerable_{pair_counter}"] = qa_pair_type == 'unanswerable'
            
            if 'options' in qa_pair:
                options_str = "\n".join([f"{key}: {value}" for option in qa_pair['options'] for key, value in option.items()])
                qa_pairs_flat[f"options_{pair_counter}"] = options_str
            
            pair_counter += 1
    
    return qa_pairs_flat

def create_task_item(item: Dict[str, Any], qa_pairs_flat: Dict[str, Any]) -> Dict[str, Any]:
    """Create a task item with flattened QA pairs."""
    task_item = {
        "data": {
            "image_file": f"/data/local-files/?d=ACL-fig/png_new/{item['image_file']}",
            "chart_id": item['chart_id'],
            "caption": item['caption'],
            "first_mention": item['first_mention'],
            "title": item['title'],
            "abstract": item['abstract'],
            "categories": item['categories'],
            "year": item['year'],
            "paper_id": item['paper_id'],
            "venue": item['venue'],
            "source_dataset": item['source_dataset'],
            "inline_reference": item.get('inline_reference', ''),
            "metadata": item.get('metadata', ''),
            "chart_type": item.get('chart_type', ''),
            "full_text": item['full_text'],
            "publisher": item['publisher'],
            "author": item['author'],
            "doi": item['doi'],
            "journal": item['journal'],
            "pdf_url": item['pdf_url']
        }
    }
    task_item["data"].update(qa_pairs_flat)
    return task_item

def process_data(original_data: pd.DataFrame, qa_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process and combine data from original and QA datasets."""
    new_json_data = []
    for item in original_data.to_dict('records'):
        chart_id = item['chart_id']
        chart_df = qa_data[qa_data['chart_id'] == chart_id]
        qa_pairs_flat = process_qa_pairs(chart_df)
        new_item = create_task_item(item, qa_pairs_flat)
        new_item["data"] = {k: v if isinstance(v, (list, dict)) else str(v) if v is not None else '' for k, v in new_item["data"].items()}
        new_json_data.append(new_item)
    return new_json_data

def save_json_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data as JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    original_data = load_pickle_data('SciVQA/data/test_run/data_sample_for_trial_annotation.pkl')
    qa_data = pd.read_pickle('SciVQA/data/test_run/qa_sample_for_trial_annotation.pkl')
    
    processed_data = process_data(original_data, qa_data)
    
    output_file = 'label_studio_data_flat_trial.json'
    save_json_data(processed_data, output_file)
    print(f"Updated JSON file has been created: {output_file}")

if __name__ == "__main__":
    main()