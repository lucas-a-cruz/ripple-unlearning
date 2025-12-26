# ruff: noqa
import copy
import json
import logging
import os
import sys

from tqdm import tqdm

# This is a hack to import from the parent directory
sys.path.append(os.path.abspath(os.path.join('third_party', 'RippleEdits', 'src')))

from build_benchmark_tests import (forward_two_hop_axis,
                                   logical_constraints_axis, making_up_axis,
                                   subject_aliasing_axis, two_hop_axis)
from query import Query
from relation import Relation
from wikidata.utils import get_aliases, get_label, subject_relation_to_targets


def safe_get_label(entity_id):
    """Safely get a label, returning the ID itself if it's not a valid Q-ID."""
    if not isinstance(entity_id, str) or not entity_id.startswith('Q'):
        return str(entity_id)
    try:
        return get_label(entity_id)
    except Exception:
        return entity_id

def query_to_dict(q: Query):
    """Safely convert a Query object to a dictionary."""
    answers = []
    target_ids = q._targets_ids if isinstance(q._targets_ids, list) else [q._targets_ids]

    flat_target_ids = []
    for item in target_ids:
        if isinstance(item, list):
            flat_target_ids.extend(item)
        else:
            flat_target_ids.append(item)

    for target_id in flat_target_ids:
        if isinstance(target_id, str) and target_id.startswith('Q'):
            answers.append({"value": get_label(target_id), "aliases": get_aliases(target_id)})
        else:
            answers.append({"value": str(target_id), "aliases": []})
    
    return {
        'prompt': q.get_query_prompt(),
        'answers': answers,
        'subject_id': q._subject_id,
        'relation': q._relation.name,
    }


def prepare_ripple_unlearning_benchmark(input_path, output_path, log_path, limit=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_path, filemode='w')
    
    # Check if we should update existing processed data or generate from scratch
    update_mode = False
    source_data = []
    
    # Check if output exists and is valid json
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
                # Check if it has content and follows the structure we expect (has 'forget' key)
                if processed_data and isinstance(processed_data, list) and len(processed_data) > 0 and 'forget' in processed_data[0]:
                    print(f"Found existing processed data at {output_path}. Switching to UPDATE mode (generating missing LG only).")
                    update_mode = True
                    source_data = processed_data
        except (json.JSONDecodeError, IndexError):
            pass
    
    if not update_mode:
        print(f"No valid existing data found (or file is empty/corrupt). Generating from scratch using raw input: {input_path}")
        if not os.path.exists(input_path):
            logging.error(f"Input file not found: {input_path}"); print(f"Error: Input file '{input_path}' not found."); return
        with open(input_path, 'r', encoding='utf-8') as f: source_data = json.load(f)

    if limit: source_data = source_data[:limit]

    final_cases = []
    
    desc = f"Updating {os.path.basename(output_path)}" if update_mode else f"Processing {os.path.basename(input_path)}"
    
    for idx, entry_data in enumerate(tqdm(source_data, desc=desc)):
        try:
            if update_mode:
                # In update mode, entry_data is the already processed case
                new_entry = entry_data # Modify in place (or copy if needed, but append handles it) 
                
                # Extract info from the 'forget' block we created previously
                forget_info = new_entry.get('forget', {})
                subject_id = forget_info.get('subject_id')
                target_id = forget_info.get('target_id')
                relation_str = forget_info.get('relation')
                
                if not (subject_id and target_id and relation_str):
                    logging.warning(f"Skipping update for item {idx}: Missing forget info.")
                    continue
                    
                relation_enum = Relation[relation_str]
                
                # Regenerate ONLY Logical Generalization
                # This fixes the missing data problem in popular.json/random.json
                # We catch errors to avoid crashing the whole update if one query fails
                try:
                    lg_tests = logical_constraints_axis(subject_id, relation_enum, target_id)
                    new_entry['Logical_Generalization'] = [t.to_dict() for t in lg_tests]
                except Exception as e:
                    logging.error(f"Failed to generate LG for case {idx}: {e}")
                    new_entry['Logical_Generalization'] = []
                
                # Other fields are preserved as is
                final_cases.append(new_entry)

            else:
                # In scratch mode, entry_data is the RAW RippleEdits entry
                case_id = entry_data.get('case_id', idx)
                if 'original_fact' not in entry_data.get('edit', {}): continue
                
                new_entry = {}
                # Copy example_type if available, or infer
                new_entry['example_type'] = entry_data.get('example_type', 'unknown')
                
                original_fact = entry_data['edit']['original_fact']
                subject_id = original_fact.get('subject_id')
                target_id = original_fact.get('target_id')
                relation_str = original_fact.get('relation')
                relation_enum = Relation[relation_str]
                
                new_forget_object = {
                    "question": relation_enum.phrase(safe_get_label(subject_id)),
                    "answer": safe_get_label(target_id),
                    "subject_id": subject_id,
                    "target_id": target_id,
                    "relation": relation_str
                }
                new_entry['forget'] = new_forget_object

                # Generate EVERYTHING from scratch (for recent.json)
                try:
                    lg_tests = logical_constraints_axis(subject_id, relation_enum, target_id)
                    new_entry['Logical_Generalization'] = [t.to_dict() for t in lg_tests]
                except Exception: new_entry['Logical_Generalization'] = []

                try:
                    c1_tests = two_hop_axis(subject_id, relation_enum, target_id)
                    new_entry['Compositionality_I'] = [t.to_dict() for t in c1_tests]
                except Exception: new_entry['Compositionality_I'] = []

                try:
                    c2_tests = forward_two_hop_axis(subject_id, relation_enum, target_id)
                    new_entry['Compositionality_II'] = [t.to_dict() for t in c2_tests]
                except Exception: new_entry['Compositionality_II'] = []

                try:
                    sa_tests = subject_aliasing_axis(subject_id, relation_enum, target_id)
                    new_entry['Subject_Aliasing'] = [t.to_dict() for t in sa_tests]
                except Exception: new_entry['Subject_Aliasing'] = []

                try:
                    rs_tests = making_up_axis(subject_id, relation_enum)
                    new_entry['Relation_Specificity'] = [t.to_dict() for t in rs_tests]
                except Exception: new_entry['Relation_Specificity'] = []
                
                final_cases.append(new_entry)

        except (KeyError, AttributeError, Exception) as e:
            logging.warning(f"Error processing item {idx}: {e}")
            continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_cases, f, ensure_ascii=False, indent=2)
    print(f"\nSuccessfully processed/updated {len(final_cases)} cases. File saved to {output_path}")

if __name__ == '__main__':
    input_dir = os.path.join('data', 'raw', 'ripple_edits_benchmark')
    output_dir = os.path.join('data', 'processed', 'ripple_unlearning_benchmark')
    
    filenames = ['popular.json', 'random.json']
    
    for filename in filenames:
        input_path = os.path.join(input_dir, filename)
        # Output filename will be the same as input, placed in the processed directory
        output_path = os.path.join(output_dir, filename)
        log_path = os.path.join(output_dir, filename.replace('.json', '_preparation.log'))

        # Check if input exists (needed for scratch mode) or output exists (for update mode)
        if os.path.exists(input_path) or os.path.exists(output_path):
            prepare_ripple_unlearning_benchmark(
                input_path=input_path, 
                output_path=output_path, 
                log_path=log_path
            )
        else:
            print(f"Files not found for {filename}, skipping.")