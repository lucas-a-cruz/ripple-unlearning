# ruff: noqa
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List

import torch
from data.unlearn import ForgetRetainDataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import register_evaluator
from .base import Evaluator
# Ensure the metrics are registered
from .metrics import ripple_metrics

logger = logging.getLogger(__name__)

class _TempDataset(Dataset):
    """A temporary dataset to wrap a list of tokenized samples."""
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

@register_evaluator("ripple_unlearning")
class RippleUnlearningEvaluator(Evaluator):
    """
    Custom evaluator for the Ripple Unlearning Benchmark.

    This evaluator performs a one-by-one evaluation of unlearning cases. For each
    case, it unlearns a single fact and then evaluates the model on a set of
    probes to measure forgetting efficacy, logical consistency, and retention
    of unrelated facts.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("ripple_unlearning", *args, **kwargs)
        self.temp_model_state_path = "temp_model_state.pt"

    def _tokenize_qa(self, question: str, answer: str) -> Dict[str, torch.Tensor]:
        """Tokenizes a question-answer pair and creates labels for fine-tuning."""
        full_text = f"{question} {answer}"
        tokenized = self.tokenizer(full_text, return_tensors="pt")
        
        question_tokens = self.tokenizer(question, return_tensors="pt")['input_ids']
        q_len = question_tokens.shape[1]

        labels = tokenized['input_ids'].clone()
        labels[0, :q_len] = -100
        
        tokenized_squeezed = {k: v.squeeze(0).to(self.model.device) for k, v in tokenized.items()}
        tokenized_squeezed['labels'] = labels.squeeze(0).to(self.model.device)
        
        return tokenized_squeezed

    def evaluate(self, model: AutoModelForCausalLM, dataset: torch.utils.data.Dataset, **kwargs):
        if self.trainer is None:
            raise ValueError("Trainer is not set for RippleUnlearningEvaluator.")

        self.model = model 
        self.tokenizer = kwargs.get("tokenizer")
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to RippleUnlearningEvaluator")

        logger.info("Saving initial clean model state...")
        torch.save(model.state_dict(), self.temp_model_state_path)
        
        all_results = defaultdict(list)
        skipped_cases = 0

        for case in tqdm(dataset, desc="Evaluating Ripple Unlearning Cases"):
            # PRE-CHECK: Verify the model knows the fact before trying to unlearn it.
            # The 'forget_efficacy' metric returns 1.0 if the model does NOT know the fact.
            # So, if efficacy is 1.0, it means the model is already ignorant, and we should skip.
            pre_check_probe = {
                "question": case["forget_request"]["question"],
                "answer": [case["forget_request"]["answer"]] # The answer here is a list for consistency with check_answers
            }
            # Note: The 'forget_efficacy' function expects a list of probes.
            # We use a temporary model state for the pre-check to not affect the actual unlearning process.
            # However, for efficiency, we can query the model directly without saving/loading for just the pre-check.
            # The model is in its initial 'clean' state here.
            
            # Create a temporary single-probe list for the efficacy check
            forget_check_probes_for_pre_check = [{
                "question": case["forget_request"]["question"],
                "answer": case["forget_probes"][0]["answer"] # Use the full answer list from forget_probes
            }]

            initial_knowledge = ripple_metrics.forget_efficacy(model, self.tokenizer, forget_check_probes_for_pre_check)
            
            # If forget_efficacy_rate is 1.0, it means the model does NOT know the fact (i.e., it's already "forgotten").
            if initial_knowledge["forget_efficacy_rate"] == 1.0:
                logger.warning(
                    f"SKIPPING case: Model does not know the fact to be unlearned. "
                    f"Fact: {case['forget_request']['question']} -> {case['forget_request']['answer']}"
                )
                skipped_cases += 1
                continue

            # If the pre-check passes (model knows the fact), proceed with unlearning.
            model.load_state_dict(torch.load(self.temp_model_state_path))

            forget_request = case["forget_request"]
            forget_sample = self._tokenize_qa(forget_request["question"], forget_request["answer"])
            forget_dataset = _TempDataset([forget_sample])

            retain_samples = [self._tokenize_qa(probe["question"], probe["answer"][0]) for probe in case["retain_probes"]]
            if not retain_samples:
                retain_samples.append(self._tokenize_qa(" ", " "))
            retain_dataset = _TempDataset(retain_samples)

            unlearn_dataset = ForgetRetainDataset(forget=forget_dataset, retain=retain_dataset, anchor="forget")

            logger.info(f"Unlearning fact: {forget_request['question']} -> {forget_request['answer']}")
            self.trainer.train(model=model, train_dataset=unlearn_dataset)
            
            with torch.no_grad():
                model.eval()

                efficacy_result = ripple_metrics.forget_efficacy(model, self.tokenizer, case.get("forget_probes", []))
                all_results["forget_efficacy_rate"].append(efficacy_result["forget_efficacy_rate"])

                inconsistency_result = ripple_metrics.logical_inconsistency(model, self.tokenizer, case.get("consistency_probes", []))
                all_results["logical_inconsistency_rate"].append(inconsistency_result["logical_inconsistency_rate"])
                
                retain_result = ripple_metrics.retain_accuracy(model, self.tokenizer, case.get("retain_probes", []))
                all_results["retain_accuracy_rate"].append(retain_result["retain_accuracy_rate"])

        if os.path.exists(self.temp_model_state_path):
            os.remove(self.temp_model_state_path)
            logger.info("Cleaned up temporary model state.")

        final_results = {}
        for key, values in all_results.items():
            if values:
                final_results[f"mean_{key}"] = sum(values) / len(values)
            else:
                final_results[f"mean_{key}"] = 0.0
        
        final_results["skipped_cases"] = skipped_cases
        total_cases = len(dataset)
        evaluated_cases = total_cases - skipped_cases
        final_results["total_cases"] = total_cases
        final_results["evaluated_cases"] = evaluated_cases

        logger.info(f"Final Aggregated Results: {final_results}")
        
        # Save final results
        output_dir = self.eval_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "ripple_unlearning_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(final_results, f, indent=4)

        logger.info(f"Ripple Unlearning evaluation summary saved to {summary_path}")

        return final_results
