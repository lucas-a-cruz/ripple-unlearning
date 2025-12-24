# ruff: noqa

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# This is a hack to import from the parent directory and the open-unlearning framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'open-unlearning')))

from src.data.ripple_dataset import RippleUnlearningDataset
from src.evals.metrics.ripple_metrics import check_answers

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_answer(model, tokenizer, question: str, device: str) -> str:
    """Generates a text answer for a given question."""
    inputs = tokenizer(question, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=25,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_ids = outputs[0][input_length:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer_text

def run_prechecks(args):
    """
    Runs pre-checks on the Ripple Unlearning benchmark to filter out cases
    that the model does not know beforehand.
    """
    logging.info(f"Loading model: {args.model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name_safe = args.model_id.replace("/", "_")
    summary_stats = {
        "model_id": args.model_id,
        "benchmarks": {},
    }

    for benchmark_name in args.benchmarks:
        input_path = os.path.join(args.input_dir, f"{benchmark_name}.jsonl")
        if not os.path.exists(input_path):
            logging.warning(f"Benchmark file not found, skipping: {input_path}")
            continue
        
        logging.info(f"Processing benchmark: {benchmark_name}")
        dataset = RippleUnlearningDataset(path=input_path)
        
        # Limit dataset for fast testing if specified
        if args.limit and args.limit > 0:
            dataset.data = dataset.data[:args.limit]
            logging.info(f"🔪 Limiting to first {args.limit} cases for testing.")

        passed_cases = []
        benchmark_summary = defaultdict(lambda: defaultdict(int))
        
        for case in tqdm(dataset, desc=f"Pre-checking {benchmark_name}"):
            case_id = case.get("case_id")
            benchmark_summary["total"]["cases"] += 1

            # 1. Check if the model knows the fact to be forgotten
            forget_probe = case["forget_probes"][0]
            forget_question = forget_probe["question"]
            expected_forget_answer = forget_probe["answer"]
            
            model_answer = get_model_answer(model, tokenizer, forget_question, device)
            knows_forget_fact = check_answers(model_answer, expected_forget_answer)

            if not knows_forget_fact:
                benchmark_summary["skipped_by_forget_check"]["cases"] += 1
                continue

            # 2. Filter consistency probes that the model already knows
            passed_consistency_probes = []
            consistency_probes = case.get("consistency_probes", [])
            
            for probe in consistency_probes:
                probe_type = probe.get("type", "unknown")
                benchmark_summary[probe_type]["total_probes"] += 1

                model_answer = get_model_answer(model, tokenizer, probe["question"], device)
                if check_answers(model_answer, probe["answer"]):
                    passed_consistency_probes.append(probe)
                    benchmark_summary[probe_type]["passed_probes"] += 1
            
            # Only include cases that have at least one valid consistency probe left
            if passed_consistency_probes:
                new_case = case.copy()
                new_case["consistency_probes"] = passed_consistency_probes
                passed_cases.append(new_case)
                benchmark_summary["final_passed_cases"]["cases"] += 1
            else:
                benchmark_summary["skipped_by_consistency_check"]["cases"] += 1

        # Save the filtered benchmark
        output_dir = os.path.join(args.output_dir, model_name_safe)
        os.makedirs(output_dir, exist_ok=True)
        filtered_output_path = os.path.join(output_dir, f"{benchmark_name}_prechecked.jsonl")
        
        with open(filtered_output_path, 'w', encoding='utf-8') as f:
            for case in passed_cases:
                f.write(json.dumps(case) + '\n')
        logging.info(f"Saved pre-checked benchmark to {filtered_output_path}")

        summary_stats["benchmarks"][benchmark_name] = benchmark_summary

    # Save the summary statistics
    summary_output_path = os.path.join(args.output_dir, model_name_safe, "precheck_summary.json")
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4)
    logging.info(f"Saved pre-check summary to {summary_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pre-checks on the Ripple Unlearning benchmark.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID.")
    parser.add_argument("--benchmarks", nargs='+', default=['popular', 'random', 'recent'], help="List of benchmarks to process.")
    parser.add_argument("--input_dir", type=str, default="data/processed/ripple_unlearning_benchmark", help="Directory containing the benchmark files.")
    parser.add_argument("--output_dir", type=str, default="data/processed/ripple_unlearning_benchmark", help="Directory to save the filtered benchmarks and summary.")
    parser.add_argument("--limit", type=int, default=25, help="Limit the number of cases to process for each benchmark for quick testing. Set to 0 for no limit.")
    
    args = parser.parse_args()
    run_prechecks(args)
