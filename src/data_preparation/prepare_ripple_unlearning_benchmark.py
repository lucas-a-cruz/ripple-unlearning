# ruff: noqa
import copy
import json
import logging
import os
import sys

from tqdm import tqdm

# This is a hack to import from the parent directory
sys.path.append(os.path.abspath(os.path.join('third_party', 'RippleEdits', 'src')))

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
    print(f"Processing {input_path}... Full logs in {log_path}")

    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}"); print(f"Error: Input file '{input_path}' not found."); return

    with open(input_path, 'r', encoding='utf-8') as f: data = json.load(f)
    if limit: data = data[:limit]

    all_unlearning_cases = []
    for idx, entry_data in enumerate(tqdm(data, desc=f"Processing {os.path.basename(input_path)}")):
        case_id = entry_data.get('case_id', idx)
        if 'original_fact' not in entry_data.get('edit', {}): continue
        
        new_entry = copy.deepcopy(entry_data)
        original_fact = new_entry.pop('edit')['original_fact']
        
        try:
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
        except (KeyError, AttributeError) as e:
            logging.warning(f"Case {case_id}: Could not process forget fact. Skipping. Error: {e}")
            continue
        
        for test_key in list(new_entry.keys()):
            if test_key not in ['Logical_Generalization', 'Compositionality_I', 'Compositionality_II', 'Subject_Aliasing', 'Relation_Specificity', 'Preservation', 'Forgetfulness']: continue
            
            updated_test_cases = []
            for test_case in new_entry.get(test_key, []):
                
                def reground_query(query, is_condition=False):
                    try:
                        if is_condition and test_key in ['Compositionality_I', 'Compositionality_II']:
                            new_cond_subject_id = original_fact.get('target_id')
                            relation_enum = Relation[query['relation']]
                            ground_truth_targets = subject_relation_to_targets(new_cond_subject_id, relation_enum.id())
                            if ground_truth_targets:
                                return query_to_dict(Query(new_cond_subject_id, relation_enum, ground_truth_targets))
                        elif query.get('query_type') == 'two_hop' and 'second_relation' in query:
                            first_hop_target_id = original_fact.get('target_id')
                            if first_hop_target_id and query['second_relation'] in Relation.__members__:
                                second_relation_enum = Relation[query['second_relation']]
                                new_target_ids = subject_relation_to_targets(first_hop_target_id, second_relation_enum.id())
                                if new_target_ids:
                                    query['answers'] = [{"value": safe_get_label(tid), "aliases": get_aliases(tid)} for tid in new_target_ids]
                                    return query
                        
                        elif test_key in ['Subject_Aliasing', 'Forgetfulness']: # Forgetfulness prompt remains as is from RippleEdits data
                            query['answers'] = [{"value": safe_get_label(original_fact.get('target_id')), "aliases": get_aliases(original_fact.get('target_id'))}]
                            return query
                        
                        elif test_key in ['Relation_Specificity', 'Preservation']:
                            return query

                    except Exception as e:
                        logging.error(f"Case {case_id}: [{test_key}] Failed to re-ground a query. Error: {e}", exc_info=True)
                    return None

                updated_test_queries = [q for q in [reground_query(query, is_condition=False) for query in test_case.get('test_queries', [])] if q]
                updated_condition_queries = [q for q in [reground_query(query, is_condition=True) for query in test_case.get('condition_queries', [])] if q]

                if updated_test_queries:
                    test_case['test_queries'] = updated_test_queries
                    test_case['condition_queries'] = updated_condition_queries
                    updated_test_cases.append(test_case)

            if updated_test_cases: new_entry[test_key] = updated_test_cases
            elif test_key in new_entry: del new_entry[test_key]
        all_unlearning_cases.append(new_entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_unlearning_cases, f, ensure_ascii=False, indent=2)
    print(f"\nSuccessfully processed {len(all_unlearning_cases)} cases. File saved to {output_path}")

if __name__ == '__main__':
    input_dir = os.path.join('data', 'raw', 'ripple_edits_benchmark')
    output_dir = os.path.join('data', 'processed', 'ripple_unlearning_benchmark')
    
    filenames = ['popular.json', 'random.json', 'recent.json']
    
    for filename in filenames:
        input_path = os.path.join(input_dir, filename)
        # Output filename will be the same as input, placed in the processed directory
        output_path = os.path.join(output_dir, filename)
        log_path = os.path.join(output_dir, filename.replace('.json', '_preparation.log'))

        if os.path.exists(input_path):
            # Process the full file by removing the `limit` parameter
            prepare_ripple_unlearning_benchmark(
                input_path=input_path, 
                output_path=output_path, 
                log_path=log_path
            )
        else:
            print(f"File not found, skipping: {input_path}")
