import os
import json
import random
import base64
from openai import OpenAI
from PIL import Image
import io
from tqdm import tqdm
import time
import re
from datetime import datetime
import argparse

class DynamicFewShotBaseline:
    def __init__(self, training_data_path, api_key=None, model="gpt-4.1-mini", debug_log_file=None, prompt_log_file=None, cleaning_log_file=None):
        # Initialize OpenAI client
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it explicitly.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.raw_training_samples = self._load_training_data(training_data_path)
        if not self.raw_training_samples:
            raise ValueError(f"Could not load training data from {training_data_path} or data is empty.")
        self.training_samples_by_qa_type = self._preprocess_training_data()
        self.debug_log_file = debug_log_file
        self.prompt_log_file = prompt_log_file
        self.cleaning_log_file = cleaning_log_file or "logs/unanswerable_cleaning_log.jsonl"

        for log_file in [self.debug_log_file, self.prompt_log_file, self.cleaning_log_file]:
            if log_file:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

    def encode_image(self, image_path):
        """
        Encode an image to base64 for API submission
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _load_training_data(self, file_path):
        """Loads training data from a JSON file."""
        if not os.path.exists(file_path):
            print(f"Error: Training data file {file_path} not found.")
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading training data from {file_path}: {e}")
            return []

    def _preprocess_training_data(self):
        """Preprocesses training data by grouping samples by qa_pair_type."""
        processed_data = {}
        for sample in self.raw_training_samples:
            qa_type = sample.get('qa_pair_type')
            if qa_type:
                if qa_type not in processed_data:
                    processed_data[qa_type] = []
                processed_data[qa_type].append(sample)
        return processed_data

    def _format_example_for_prompt(self, sample, example_number, include_qa_type: bool = True):
        """Format example."""
        question_text = sample.get('question', '')
        answer_text = sample.get('answer', '')
        answer_options_list = sample.get('answer_options', [])
        current_qa_pair_type = sample.get('qa_pair_type', '')
        figure_type = sample.get('figure_type', '')
        caption = sample.get('caption', '')

        # Normalize answer
        if current_qa_pair_type in ["closed-ended finite answer set binary visual", "closed-ended finite answer set binary non-visual"]:
            if answer_text.lower() == "yes":
                answer_text = "Yes"
            elif answer_text.lower() == "no":
                answer_text = "No"
        elif current_qa_pair_type in ["closed-ended finite answer set non-binary visual", "closed-ended finite answer set non-binary non-visual"]:
            if "," in answer_text:
                answer_text = re.sub(r',\s*', ',', answer_text)

        # Header
        example_str = f"Example {example_number}:\n"
        # Meta
        example_str += f"Figure Caption: {caption}\n"
        example_str += f"Figure Type: {figure_type}\n"
        if include_qa_type:
            example_str += f"Question Type: {current_qa_pair_type}\n"
        example_str += f"Question: {question_text}\n"
        
        # Options
        if answer_options_list and len(answer_options_list) > 0 and current_qa_pair_type in ["closed-ended finite answer set non-binary visual", "closed-ended finite answer set non-binary non-visual"]:
            example_str += "Answer Options:\n"
            for option_set in answer_options_list:
                for key, value in option_set.items():
                    if value: 
                        example_str += f"{key}: {value}\n"
        
        # Correct
        example_str += f"\nCorrect answer: {answer_text}\n"
        
        return example_str

    def _get_dynamic_examples(self, current_qa_pair_type, current_figure_type, num_examples=5, mix_random_nonmatching: bool = False, include_qa_type: bool = True):
        """Get few-shot examples."""
        if not mix_random_nonmatching:
            if current_qa_pair_type not in self.training_samples_by_qa_type:
                return ""
            candidate_samples = self.training_samples_by_qa_type[current_qa_pair_type]
            if not candidate_samples:
                return ""

            preferred_samples = [s for s in candidate_samples if s.get('figure_type') == current_figure_type]
            other_samples = [s for s in candidate_samples if s.get('figure_type') != current_figure_type]

            selected_samples = []
            if preferred_samples:
                take_from_preferred = min(len(preferred_samples), num_examples)
                selected_samples.extend(random.sample(preferred_samples, take_from_preferred))

            remaining_needed = num_examples - len(selected_samples)
            if remaining_needed > 0 and other_samples:
                take_from_other = min(len(other_samples), remaining_needed)
                selected_samples.extend(random.sample(other_samples, take_from_other))
        else:
            # logic for unanswerable
            unanswerable_samples_all = self.training_samples_by_qa_type.get("unanswerable", [])
            # split by figure-type priority
            ua_pref = [s for s in unanswerable_samples_all if s.get("figure_type") == current_figure_type]
            ua_other = [s for s in unanswerable_samples_all if s.get("figure_type") != current_figure_type]

            # collect all other qa-types
            other_type_samples_all = []
            for qa_type, samples in self.training_samples_by_qa_type.items():
                if qa_type != "unanswerable":
                    other_type_samples_all.extend(samples)

            ot_pref = [s for s in other_type_samples_all if s.get("figure_type") == current_figure_type]
            ot_other = [s for s in other_type_samples_all if s.get("figure_type") != current_figure_type]

            selected_samples = []

            # 2 unanswerable
            ua_needed = min(2, len(ua_pref) + len(ua_other))
            if ua_pref:
                take = min(len(ua_pref), ua_needed)
                selected_samples.extend(random.sample(ua_pref, take))
                ua_needed -= take
            if ua_needed > 0 and ua_other:
                selected_samples.extend(random.sample(ua_other, min(ua_needed, len(ua_other))))

            # remaining slots with other types
            remaining_needed = num_examples - len(selected_samples)
            if remaining_needed > 0:
                pool = []
                if ot_pref:
                    take = min(len(ot_pref), remaining_needed)
                    pool.extend(random.sample(ot_pref, take))
                    remaining_needed -= take
                if remaining_needed > 0 and ot_other:
                    pool.extend(random.sample(ot_other, min(remaining_needed, len(ot_other))))
                selected_samples.extend(pool)

        if not selected_samples:
            return ""
            
        random.shuffle(selected_samples)

        # Format
        example_prompts = []
        for i, sample in enumerate(selected_samples):
            example_prompts.append(self._format_example_for_prompt(sample, i + 1, include_qa_type=include_qa_type))
        
        return "\n".join(example_prompts)

    def generate_prompt(self, question, caption, figure_type, answer_options=None, qa_pair_type=None, hide_qa_pair_type: bool = False):
        """Build prompts."""
        
        # Sys prompt
        base_system_prompt = """You are an expert scientific figure analyst specializing in academic publications.
Your task is to answer questions about scientific figures and their captions accurately and concisely.
Answer the given question based *solely* on the information visible in the figure and its provided caption.

The user message will include a 'Question Type'. Adhere strictly to the following rules for formatting your response based on the question type:

- For 'closed-ended finite answer set binary visual' or 'closed-ended finite answer set binary non-visual': 
  - Respond ONLY with 'Yes' or 'No'. 
  - Do NOT add any other text, explanations, or punctuation.
  - Your entire response must be exactly one word: either 'Yes' or 'No'.

- For 'closed-ended finite answer set non-binary visual' or 'closed-ended finite answer set non-binary non-visual': 
  - Identify the correct option(s) from the provided 'Answer Options'.
  - Respond ONLY with the letter(s) of the correct option(s) as listed.
  - For a single correct option, provide only its letter (e.g., 'B').
  - For multiple correct options, list ALL correct letters separated by commas with NO SPACES (e.g., 'A,C,D').
  - Ensure ALL correct options are listed and NO incorrect ones.
  - Do NOT add any other text, explanations, or surrounding punctuation.

- For 'closed-ended infinite answer set visual' or 'closed-ended infinite answer set non-visual': 
  - Provide a brief, direct answer.
  - This answer must be a value, a short phrase, a specific name, a label, or a list of values read directly from the figure or caption.
  - **For numerical values:** Read values as precisely as possible from the graph axes, data points, or labels. Include units ONLY if they appear in the figure.
  - **For non-numerical values:** Reproduce them EXACTLY as they appear in the figure or caption.
  - Do NOT add any introductory phrases, explanations, or surrounding text.

- For 'unanswerable': 
  - Respond ONLY with the exact phrase: 'It is not possible to answer this question based only on the provided data.'
  - Do NOT add any other text.

IMPORTANT: Your response should ONLY contain the answer in the correct format as specified above - nothing else.
Do NOT include any additional text, explanations, comments, or contextual information.
Your answer must be based solely on the information visible in the figure and its provided caption.
"""
 
        hide_qa = qa_pair_type == "unanswerable"
        dynamic_examples_str = self._get_dynamic_examples(
            qa_pair_type,
            figure_type,
            num_examples=5,
            mix_random_nonmatching=hide_qa,
            include_qa_type=not hide_qa,
        )

        # Add examples to the system prompt
        full_system_prompt = base_system_prompt
        if dynamic_examples_str:
            full_system_prompt += (
                "\n\nBelow are examples of questions and answers similar to what you will receive. "
                "Study these examples carefully to understand the expected answer format. "
                "Your question will be in the user message after these examples:\n\n"
            ) + dynamic_examples_str
        
        # Type rules
        if qa_pair_type:
            if qa_pair_type in ["closed-ended finite answer set binary visual", "closed-ended finite answer set binary non-visual"]:
                full_system_prompt += "\n\nREMEMBER: Your entire answer must be EXACTLY 'Yes' or 'No' - nothing more, nothing less."
            elif qa_pair_type in ["closed-ended finite answer set non-binary visual", "closed-ended finite answer set non-binary non-visual"] and answer_options:
                full_system_prompt += "\n\nREMEMBER: Your entire answer must be ONLY the letter(s) of the correct option(s) - e.g., 'A' or 'B,D'."
            elif qa_pair_type in ["closed-ended infinite answer set visual", "closed-ended infinite answer set non-visual"]:
                full_system_prompt += "\n\nREMEMBER: Your answer must be concise and direct, with no explanatory text."
            elif qa_pair_type == "unanswerable":
                full_system_prompt += "\n\nREMEMBER: Decide if the question is unanswerable based on the figure and caption. If it is, respond with 'It is not possible to answer this question based only on the provided data.'. If it is not, respond with the correct answer."

        # Build user prompt
        user_prompt_text = f"Figure Caption: {caption}\nFigure Type: {figure_type}\n"
        
        if qa_pair_type and not hide_qa_pair_type:
            user_prompt_text += f"Question Type: {qa_pair_type}\n"
        
        user_prompt_text += f"Question: {question}\n"
        
        if answer_options and len(answer_options) > 0:
            options_text_build = "\nAnswer Options:"
            for option_list_item in answer_options: 
                for key, value in option_list_item.items():
                    if value:
                        options_text_build += f"\n{key}: {value}"
            user_prompt_text += options_text_build

        return full_system_prompt, user_prompt_text

    def _log_cleaning_event(self, instance_id, raw_answer, cleaned_answer, qa_pair_type):
        """Write a json line recording a cleaning operation (only used for unanswerable)."""
        if not self.cleaning_log_file:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "instance_id": instance_id,
            "qa_pair_type": qa_pair_type,
            "raw_answer": raw_answer,
            "cleaned_answer": cleaned_answer,
        }
        try:
            with open(self.cleaning_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Warning: could not write cleaning log: {e}")

    def _is_answer_indicating_unanswerable(self, answer: str, threshold: float = 0.4) -> bool:
        """Check if answer is similar to standard unanswerable answer using token overlap."""
        # Normalize both texts: lowercase, remove punctuation
        answer = answer.lower().strip()
        standard = "it is not possible to answer this question based only on the provided data"
        
        answer_tokens = set(answer.split())
        standard_tokens = set(standard.split())
        
        overlap = len(answer_tokens.intersection(standard_tokens))
        similarity = overlap / len(standard_tokens)
        
        return similarity >= threshold

    def answer_question(self, image_path, question, caption, figure_type, answer_options=None, qa_pair_type=None, instance_id=None):
        """Answers a question using GPT-4.1-mini with dynamic few-shot examples and logs prompts/responses if configured."""
        base64_image = self.encode_image(image_path)

        hide_qa = qa_pair_type == "unanswerable"
        full_system_prompt, user_prompt_text = self.generate_prompt(
            question, caption, figure_type, answer_options, qa_pair_type, hide_qa_pair_type=hide_qa
        )
        # Format prompt output for logs and terminal
        if self.prompt_log_file:
            prompt_output = f"\n============ DYNAMICALLY GENERATED PROMPTS ============\n"
            prompt_output += f"\n--- SYSTEM PROMPT ---\n"
            prompt_output += full_system_prompt
            prompt_output += f"\n\n--- USER PROMPT ---\n"
            prompt_output += user_prompt_text
            prompt_output += f"\n\n============ END OF PROMPTS ============\n"
        # Print generated prompts
            print(prompt_output)
            
            timestamp = datetime.utcnow().isoformat() + "Z"
            with open(self.prompt_log_file, 'a', encoding='utf-8') as f_prompt:
                f_prompt.write(f"\n\n=== PROMPT LOG [{timestamp}] ===\n")
                f_prompt.write(f"Image: {image_path}\n")
                f_prompt.write(f"Question: {question}\n")
                f_prompt.write(f"Question Type: {qa_pair_type}\n")
                f_prompt.write(prompt_output)

        log_entry = None
        if self.debug_log_file:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "image_path": image_path,
                "question": question,
                "qa_pair_type": qa_pair_type,
                "figure_type": figure_type,
                "system_prompt": full_system_prompt,
                "user_prompt": user_prompt_text,
                "model_response": None
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.0
            )
            
            predicted_raw = response.choices[0].message.content.strip()
            predicted_answer = predicted_raw
            # Post-process predicted answer
            if qa_pair_type:
                if qa_pair_type in ["closed-ended finite answer set binary visual", "closed-ended finite answer set binary non-visual"]:
                    if re.search(r'yes', predicted_answer.lower()):
                        predicted_answer = "Yes"
                    elif re.search(r'no', predicted_answer.lower()):
                        predicted_answer = "No"
                
                elif qa_pair_type in ["closed-ended finite answer set non-binary visual", "closed-ended finite answer set non-binary non-visual"] and answer_options:
                    option_match = re.search(r'([A-D](,[A-D])*)', predicted_answer.replace(" ", ""))
                    if option_match:
                        predicted_answer = option_match.group(1)
                
                elif qa_pair_type in ["closed-ended infinite answer set visual", "closed-ended infinite answer set non-visual"]:
                    intro_pattern = re.compile(r'^(the\s+answer\s+is\s+|the\s+value\s+is\s+)', re.IGNORECASE)
                    predicted_answer = intro_pattern.sub('', predicted_answer)
                    predicted_answer = re.sub(r',\s+', ',', predicted_answer)
                    predicted_answer = predicted_answer.strip()
                
                elif qa_pair_type == "unanswerable":
                    if self._is_answer_indicating_unanswerable(predicted_raw):
                        predicted_answer = "It is not possible to answer this question based only on the provided data."
            
            if qa_pair_type == "unanswerable":
                self._log_cleaning_event(instance_id, predicted_raw, predicted_answer, qa_pair_type)
            
            if log_entry and self.debug_log_file:
                log_entry["model_response"] = predicted_answer
                with open(self.debug_log_file, 'a', encoding='utf-8') as f_log:
                    f_log.write(json.dumps(log_entry) + '\n')
            return predicted_answer
        
        except Exception as e:
            error_message = f"Error calling OpenAI API: {e}"
            print(error_message)
            if log_entry and self.debug_log_file:
                log_entry["model_response"] = f"ERROR: {error_message}"
                with open(self.debug_log_file, 'a', encoding='utf-8') as f_log:
                    f_log.write(json.dumps(log_entry) + '\n')
            return None

    def process_dataset(self, input_file, output_file, images_dir, limit=None, unanswerable_only=False):
        """
        Process a dataset file and generate answers for all questions
        
        Args:
            input_file: Path to the input JSON file with questions
            output_file: Path to save the output JSON file with answers
            images_dir: Directory containing the figure images
            limit: Limit the number of samples to process (for testing)
            unanswerable_only: If True, only recompute unanswerable questions and take other answers from backup file
        """
        # Load the dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        backup_predictions = {}
        if unanswerable_only:
            backup_file = "data/processed/test_predictions_backup.json"
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                    for item in backup_data:
                        key = f"{item['question']}_{item['image_file']}"
                        backup_predictions[key] = item['predicted_answer']
                print(f"Loaded {len(backup_predictions)} backup predictions from {backup_file}")
            except Exception as e:
                print(f"Error loading backup predictions: {e}")
                return None
        
        if limit:
            data = data[:limit]
        
        results = []
        unanswerable_count = 0
        
        for sample in tqdm(data, desc="Processing samples"):
            qa_pair_type = sample['qa_pair_type']
            
            if unanswerable_only and qa_pair_type != "unanswerable":
                key = f"{sample['question']}_{sample['image_file']}"
                if key in backup_predictions:
                    result = sample.copy()
                    result['predicted_answer'] = backup_predictions[key]
                    results.append(result)
                else:
                    print(f"Warning: No backup prediction found for question: {sample['question'][:50]}...")
                continue
            
            if qa_pair_type == "unanswerable":
                unanswerable_count += 1
            
            image_file = sample['image_file']
            split_name = "train" if "train" in input_file else "validation" if "validation" in input_file else "test"
            image_path = os.path.join(images_dir, split_name, image_file)
            
            question = sample['question']
            caption = sample['caption']
            figure_type = sample['figure_type']
            answer_options = sample['answer_options'] if 'answer_options' in sample else []
            
            # Get answer from model
            predicted_answer = self.answer_question(
                image_path,
                question,
                caption,
                figure_type,
                answer_options,
                qa_pair_type,
                instance_id=sample.get("id")
            )
            
            # Create result object
            result = sample.copy()
            result['predicted_answer'] = predicted_answer
            results.append(result)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        if unanswerable_only:
            print(f"Processed {unanswerable_count} unanswerable questions out of {len(results)} total samples.")
        else:
            print(f"Processed {len(results)} samples.")
        print(f"Results saved to {output_file}")
        
        return results

def main():
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return
    
    # Initialize the model
    model = DynamicFewShotBaseline(
        training_data_path="data/raw/train.json",
        debug_log_file="logs/dynamic_few_shot_debug.log",
        prompt_log_file="logs/prompt_examples.txt",
        cleaning_log_file="logs/unanswerable_cleaning_log.jsonl"
    )
    
    print("Starting test dataset processing...")
    print("All generated prompts will be saved in 'logs/prompt_examples.txt'.")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process dataset with dynamic few-shot baseline')
    parser.add_argument('--unanswerable-only', action='store_true',
                      help='Only recompute unanswerable questions and take other answers from backup file')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit the number of samples to process (for testing)')
    args = parser.parse_args()
    
    model.process_dataset(
        input_file="data/raw/test.json",
        output_file="data/processed/test_predictions.json",
        images_dir="data/raw/images",
        limit=args.limit,
        unanswerable_only=args.unanswerable_only
    )
    
    print("Test dataset processing completed!")

if __name__ == "__main__":
    main() 