# ruff: noqa
import json
import logging
import os
import sys
from collections import defaultdict

from tqdm import tqdm

# This is a hack to import from the parent directory
sys.path.append(os.path.abspath(os.path.join('third_party', 'RippleEdits', 'src')))
from wikidata.utils import get_aliases, get_label

# --- Copied from prepare_ripple_unlearning_benchmark.py to avoid cross-imports ---
def safe_get_label(entity_id):
    """Safely get a label, returning the ID itself if it's not a valid Q-ID."""
    if not isinstance(entity_id, str) or not entity_id.startswith('Q'):
        return str(entity_id)
    try:
        return get_label(entity_id)
    except Exception:
        return entity_id
# --- End of copied code ---

CONSISTENCY_PROBE_TYPES = [
    "Logical_Generalization",
    "Compositionality_I",
    "Compositionality_II",
    "Subject_Aliasing",
    "Forgetfulness", 
]
RETAIN_PROBE_TYPES = ["Relation_Specificity", "Preservation"]


def build_unlearning_benchmark(input_path, output_path, log_path, limit=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_path, filemode='w')
    print(f"Building final benchmark from {input_path}... Full logs in {log_path}")
    
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        print(f"Error: Input file '{input_path}' not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    processed_cases = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in tqdm(data, desc=f"Building {os.path.basename(output_path)}"):
            
            raw_forget_request = entry.get('forget')
            if not raw_forget_request:
                logging.warning(f"Skipping entry {entry.get('case_id')} due to missing 'forget' object.")
                continue

            final_forget_request = {
                "question": raw_forget_request["question"],
                "answer": raw_forget_request["answer"]
            }
            
            def transform_probes(probe_category_list, probe_type_key):
                final_probes = []
                for test_case_dict in probe_category_list:
                    for query_dict in test_case_dict.get('test_queries', []):
                        question = query_dict.get('prompt')
                        
                        # Collect the primary value and all aliases from all answer objects
                        all_possible_answers = []
                        for ans_item in query_dict.get("answers", []):
                            if ans_item.get("value"):
                                all_possible_answers.append(ans_item["value"])
                            if ans_item.get("aliases"):
                                all_possible_answers.extend(ans_item["aliases"])
                        
                        # Create a unique list of answers
                        unique_answers = list(dict.fromkeys(all_possible_answers))
                        
                        if question and unique_answers:
                            final_probes.append({
                                "question": question,
                                "answer": unique_answers,
                                "type": probe_type_key
                            })
                return final_probes

            main_answer = final_forget_request["answer"]
            target_id = raw_forget_request.get("target_id")
            aliases = get_aliases(target_id) if (target_id and isinstance(target_id, str) and target_id.startswith('Q')) else []
            all_answers = list(dict.fromkeys([main_answer] + aliases))

            forget_probes = [{
                "question": final_forget_request["question"],
                "answer": all_answers,
                "type": "Forgetfulness" # Assign a type for consistency
            }]
            
            consistency_probes = []
            for key in CONSISTENCY_PROBE_TYPES:
                if key in entry:
                    consistency_probes.extend(transform_probes(entry[key], key))
            
            retain_probes = []
            for key in RETAIN_PROBE_TYPES:
                if key in entry:
                    retain_probes.extend(transform_probes(entry[key], key))
            
            final_record = {
                "example_type": entry.get("example_type", "unknown"),
                "forget_request": final_forget_request,
                "forget_probes": forget_probes,
                "consistency_probes": consistency_probes,
                "retain_probes": retain_probes
            }
            
            f.write(json.dumps(final_record) + '\n')
            processed_cases += 1
            
    print(f"\nSuccessfully built {processed_cases} cases.")
    print(f"Final benchmark file saved to {output_path}")

if __name__ == '__main__':
    base_dir = os.path.join('data', 'processed', 'ripple_unlearning_benchmark')
    
    filenames = ['popular.json', 'random.json', 'recent.json']
    
    for filename in filenames:
        input_path = os.path.join(base_dir, filename)
        output_filename = filename.replace('.json', '.jsonl')
        output_path = os.path.join(base_dir, output_filename)
        log_path = os.path.join(base_dir, output_filename.replace('.jsonl', '_build.log'))
        
        if os.path.exists(input_path):
            build_unlearning_benchmark(input_path, output_path, log_path)
        else:
            print(f"Intermediate file not found, skipping: {input_path}")
            print("Please run `prepare_ripple_unlearning_benchmark.py` first.")