# ruff: noqa
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List

import torch
import evaluate
from data.collators import DataCollatorForSupervisedDataset
from data.ripple_dataset import RippleUnlearningDataset
from data.unlearn import ForgetRetainDataset
from torch.utils.data import Dataset
from tqdm import tqdm
from trainer import load_trainer
from transformers import AutoModelForCausalLM

from .base import Evaluator
# Ensure the metrics are registered
from evals.metrics.utils import evaluate_probability
from evals.metrics import ripple_metrics
from evals.metrics.ripple_metrics import check_answers

logger = logging.getLogger(__name__)

class RippleUnlearningEvaluator(Evaluator):
    """
    Custom evaluator for the Ripple Unlearning Benchmark.
    This evaluator performs a one-by-one evaluation of unlearning cases.
    """
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("ripple_unlearning", eval_cfg=eval_cfg, **kwargs)
        self.trainer = None
        self.temp_model_state_path = "temp_model_state.pt"

    def _get_prob_of_answer(self, model, question, answer) -> float:
        """
        Calculates the probability of a model generating a specific answer for a given question.
        Reuses the framework's `evaluate_probability` utility.
        """
        if not answer or not isinstance(answer, str):
            return 0.0

        tokenized_pair = self._tokenize_qa(self.tokenizer, self.template_args, question, answer)
        
        # Create a batch of size 1
        batch = {
            "input_ids": tokenized_pair['input_ids'].unsqueeze(0),
            "attention_mask": tokenized_pair['attention_mask'].unsqueeze(0),
            "labels": tokenized_pair['labels'].unsqueeze(0)
        }
        
        # Move batch to model's device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            # evaluate_probability expects a batch and returns a list of dicts
            prob_results = evaluate_probability(model, batch)
        
        if prob_results and prob_results[0] and "prob" in prob_results[0]:
            return prob_results[0]["prob"]
        
        return 0.0

    @staticmethod
    def _tokenize_qa(tokenizer, template_args, question: str, answer: str) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a question-answer pair using the model's chat template.
        Returns a dictionary of CPU tensors.
        """
        chat = []
        if template_args.get("apply_chat_template"):
            system_prompt = template_args.get("system_prompt")
            if system_prompt:
                chat.append({"role": "system", "content": system_prompt})
            chat.append({"role": "user", "content": question})
            chat.append({"role": "assistant", "content": answer})
            
            # Tokenize the full chat to get input_ids and attention_mask
            tokenized_ids = tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=False
            )
            
            # Tokenize up to the user part to find the length of the prompt for masking labels
            prompt_ids = tokenizer.apply_chat_template(
                chat[:-1], tokenize=True, add_generation_prompt=True
            )
            q_len = len(prompt_ids)

            labels = list(tokenized_ids)
            labels[:q_len] = [-100] * q_len
            
            item = {
                "input_ids": tokenized_ids,
                "attention_mask": [1] * len(tokenized_ids),
                "labels": labels
            }
        else: # Fallback for non-chat models
            full_text = f"{question} {answer}"
            tokenized = tokenizer(full_text, add_special_tokens=False)
            question_tokens = tokenizer(question, add_special_tokens=False)['input_ids']
            q_len = len(question_tokens)

            labels = list(tokenized['input_ids'])
            labels[:q_len] = [-100] * q_len
            
            item = {
                "input_ids": tokenized['input_ids'],
                "attention_mask": tokenized['attention_mask'],
                "labels": labels
            }
        
        return {k: torch.tensor(v) for k, v in item.items()}

    @staticmethod
    def _get_answer_for_probe(model, tokenizer, template_args, probe, log_prompt: bool = False) -> str:
        """Generates a text answer for a given probe question, applying chat template."""
        if not probe or "question" not in probe:
            return "Invalid Probe"
            
        question = probe["question"]
        
        chat = []
        if template_args.get("apply_chat_template"):
            system_prompt = template_args.get("system_prompt")
            if system_prompt:
                chat.append({"role": "system", "content": system_prompt})
            chat.append({"role": "user", "content": question})
            
            if log_prompt:
                # Log the exact prompt string that will be tokenized
                prompt_for_logging = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                logger.info(f"\n[PROMPT FED TO MODEL]:\n---\n{prompt_for_logging}\n---")

            input_ids = tokenizer.apply_chat_template(
                chat, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(model.device)
            inputs = {'input_ids': input_ids}
            input_length = inputs['input_ids'].shape[1]
        else: # Fallback for non-chat models
            if log_prompt:
                logger.info(f"\n[PROMPT FED TO MODEL]:\n---\n{question}\n---")
            inputs = tokenizer(question, return_tensors="pt").to(model.device)
            input_length = inputs['input_ids'].shape[1]

        stop_ids = [tokenizer.eos_token_id]
        newline_token_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_token_ids) <= 2:
            stop_ids.extend(newline_token_ids)

        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids
        )
        generated_ids = outputs[0][input_length:]
        answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer_text

    def evaluate(self, model: AutoModelForCausalLM, **kwargs):
        self.model = model
        self.tokenizer = kwargs.get("tokenizer")
        self.template_args = kwargs.get("template_args")
        if self.tokenizer is None or self.template_args is None:
            raise ValueError("Tokenizer and template_args must be provided to RippleUnlearningEvaluator")

        rouge_scorer = evaluate.load('rouge')
        
        dataset_path = self.eval_cfg.data.ripple_unlearning.args.path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please provide a valid path via `data.ripple_unlearning.args.path`.")
        
        dataset = RippleUnlearningDataset(path=dataset_path)
        logger.info(f"Loaded Ripple Unlearning dataset with {len(dataset)} cases from {dataset_path}")

        # Limit dataset to 25 samples for quick testing
        dataset.data = dataset.data[:25]
        logger.info(f"🔪 Limiting evaluation to the first 25 cases for testing.")

        torch.save(model.state_dict(), self.temp_model_state_path)
        
        aggregated_results = defaultdict(list)
        detailed_results = []
        # Pre-check is no longer performed here, so no cases will be skipped.
        skipped_cases = 0

        for case in tqdm(dataset, desc="Evaluating Ripple Unlearning Cases"):
            case_result = {
                "case_id": case.get("case_id"),
                "metadata": case.get("metadata"),
                "passed_pre_check": True,  # Assumed true for a pre-filtered benchmark
                "probes": []
            }

            # --- Get pre-unlearning answers for all relevant probes ---
            probes_to_log = {
                "Forget": case["forget_probes"][0] if case.get("forget_probes") else None,
                "Consistency": case["consistency_probes"][0] if case.get("consistency_probes") else None,
                "Retain": case["retain_probes"][0] if case.get("retain_probes") else None,
            }
            clean_answers = {}
            for name, probe in probes_to_log.items():
                if probe:
                    text_answer = self._get_answer_for_probe(model, self.tokenizer, self.template_args, probe)
                    prob_answer = self._get_prob_of_answer(model, probe["question"], probe.get("answer"))
                    rouge_result = rouge_scorer.compute(
                        predictions=[text_answer],
                        references=[probe.get("answer","")],
                        use_stemmer=True
                    )
                    clean_answers[name] = {
                        "text": text_answer,
                        "prob": prob_answer,
                        "rouge-l": rouge_result['rougeL']
                    }


            # --- Unlearning Step ---
            model.load_state_dict(torch.load(self.temp_model_state_path))
            
            # Re-initialize the trainer to prevent state accumulation across iterations
            trainer_cfg = self.eval_cfg.get("trainer")
            data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
            self.trainer, trainer_args = load_trainer(
                trainer_cfg=trainer_cfg, model=model, train_dataset=[], data_collator=data_collator
            )
            logger.info(f"Loaded Unlearning Trainer: {self.trainer.__class__.__name__}")
            trainer_args.remove_unused_columns = False
            self.trainer.args = trainer_args

            forget_request = case["forget_request"]
            forget_dataset = [self._tokenize_qa(self.tokenizer, self.template_args, forget_request["question"], forget_request["answer"])]
            
            retain_samples = [self._tokenize_qa(self.tokenizer, self.template_args, p["question"], p["answer"][0]) for p in case.get("retain_probes", []) if p.get("answer")]
            if not retain_samples:
                retain_samples.append(self._tokenize_qa(self.tokenizer, self.template_args, " ", " ")) # Dummy sample if no retain probes

            unlearn_dataset = ForgetRetainDataset(forget=forget_dataset, retain=retain_samples, anchor="forget")
            self.trainer.train_dataset = unlearn_dataset
            self.trainer.train()
            
            # --- Evaluation Step ---
            with torch.no_grad():
                model.eval()

                # --- Qualitative and Detailed Logging ---
                logger.info("\n" + "="*80)
                logger.info(f"📊 ANALYSIS FOR CASE: {case.get('case_id', 'N/A')}")
                logger.info("=" * 80)
                
                # Process Forget Probes
                forget_probes = case.get("forget_probes", [])
                if forget_probes:
                    probe_data = forget_probes[0]
                    unlearned_answer = self._get_answer_for_probe(model, self.tokenizer, self.template_args, probe_data)
                    unlearned_prob = self._get_prob_of_answer(model, probe_data["question"], probe_data.get("answer"))
                    unlearned_rouge_result = rouge_scorer.compute(predictions=[unlearned_answer], references=[probe_data.get("answer", "")], use_stemmer=True)
                    unlearned_rouge_l = unlearned_rouge_result['rougeL']
                    
                    has_answer = check_answers(unlearned_answer, probe_data['answer'])
                    did_forget = not has_answer
                    
                    aggregated_results["forget_efficacy_rate"].append(1.0 if did_forget else 0.0)
                    aggregated_results["clean_forget_prob"].append(clean_answers.get("Forget", {}).get("prob", 0.0))
                    aggregated_results["unlearned_forget_prob"].append(unlearned_prob)
                    aggregated_results["clean_forget_rouge_l"].append(clean_answers.get("Forget", {}).get("rouge-l", 0.0))
                    aggregated_results["unlearned_forget_rouge_l"].append(unlearned_rouge_l)

                    case_result["probes"].append({
                        "type": "forget",
                        "question": probe_data['question'],
                        "expected_answer": probe_data['answer'],
                        "clean_model_answer": clean_answers.get("Forget", {}).get("text"),
                        "unlearned_model_answer": unlearned_answer,
                        "evaluation": {
                            "did_forget": did_forget,
                            "clean_prob": clean_answers.get("Forget", {}).get("prob"),
                            "unlearned_prob": unlearned_prob,
                            "clean_rouge_l": clean_answers.get("Forget", {}).get("rouge-l"),
                            "unlearned_rouge_l": unlearned_rouge_l
                        }
                    })
                    logger.info(f"🔹 Probe: Forget -> {'✅ FORGOT' if did_forget else '❌ FAILED TO FORGET'}")
                    logger.info(f"  Question: {probe_data['question']}")
                    logger.info(f"  Clean Model Answer: {clean_answers.get('Forget', {}).get('text')} (Prob: {clean_answers.get('Forget', {}).get('prob', 0.0):.4f}, ROUGE-L: {clean_answers.get('Forget', {}).get('rouge-l', 0.0):.4f})")
                    logger.info(f"  Unlearned Model Answer: {unlearned_answer} (Prob: {unlearned_prob:.4f}, ROUGE-L: {unlearned_rouge_l:.4f})")

                # Process Consistency Probes
                consistency_probes = case.get("consistency_probes", [])
                if consistency_probes:
                    probe_data = consistency_probes[0]
                    unlearned_answer = self._get_answer_for_probe(model, self.tokenizer, self.template_args, probe_data)
                    unlearned_prob = self._get_prob_of_answer(model, probe_data["question"], probe_data.get("answer"))
                    unlearned_rouge_result = rouge_scorer.compute(predictions=[unlearned_answer], references=[probe_data.get("answer", "")], use_stemmer=True)
                    unlearned_rouge_l = unlearned_rouge_result['rougeL']
                    
                    has_answer = check_answers(unlearned_answer, probe_data['answer'])
                    is_consistent = not has_answer

                    aggregated_results["logical_inconsistency_rate"].append(0.0 if is_consistent else 1.0)
                    aggregated_results["clean_consistency_prob"].append(clean_answers.get("Consistency", {}).get("prob", 0.0))
                    aggregated_results["unlearned_consistency_prob"].append(unlearned_prob)
                    aggregated_results["clean_consistency_rouge_l"].append(clean_answers.get("Consistency", {}).get("rouge-l", 0.0))
                    aggregated_results["unlearned_consistency_rouge_l"].append(unlearned_rouge_l)
                    
                    case_result["probes"].append({
                        "type": "consistency",
                        "question": probe_data['question'],
                        "expected_answer": probe_data['answer'],
                        "clean_model_answer": clean_answers.get("Consistency", {}).get("text"),
                        "unlearned_model_answer": unlearned_answer,
                        "evaluation": {
                            "is_consistent": is_consistent,
                            "clean_prob": clean_answers.get("Consistency", {}).get("prob"),
                            "unlearned_prob": unlearned_prob,
                            "clean_rouge_l": clean_answers.get("Consistency", {}).get("rouge-l"),
                            "unlearned_rouge_l": unlearned_rouge_l
                        }
                    })
                    logger.info(f"🔹 Probe: Consistency -> {'✅ CONSISTENT' if is_consistent else '❌ INCONSISTENT'}")
                    logger.info(f"  Question: {probe_data['question']}")
                    logger.info(f"  Clean Model Answer: {clean_answers.get('Consistency', {}).get('text')} (Prob: {clean_answers.get('Consistency', {}).get('prob', 0.0):.4f}, ROUGE-L: {clean_answers.get('Consistency', {}).get('rouge-l', 0.0):.4f})")
                    logger.info(f"  Unlearned Model Answer: {unlearned_answer} (Prob: {unlearned_prob:.4f}, ROUGE-L: {unlearned_rouge_l:.4f})")
                
                # Process Retain Probes
                retain_probes = case.get("retain_probes", [])
                if retain_probes:
                    probe_data = retain_probes[0]
                    unlearned_answer = self._get_answer_for_probe(model, self.tokenizer, self.template_args, probe_data)
                    unlearned_prob = self._get_prob_of_answer(model, probe_data["question"], probe_data.get("answer"))
                    unlearned_rouge_result = rouge_scorer.compute(predictions=[unlearned_answer], references=[probe_data.get("answer", "")], use_stemmer=True)
                    unlearned_rouge_l = unlearned_rouge_result['rougeL']

                    has_answer = check_answers(unlearned_answer, probe_data['answer'])
                    did_retain = has_answer

                    aggregated_results["retain_accuracy_rate"].append(1.0 if did_retain else 0.0)
                    aggregated_results["clean_retain_prob"].append(clean_answers.get("Retain", {}).get("prob", 0.0))
                    aggregated_results["unlearned_retain_prob"].append(unlearned_prob)
                    aggregated_results["clean_retain_rouge_l"].append(clean_answers.get("Retain", {}).get("rouge-l", 0.0))
                    aggregated_results["unlearned_retain_rouge_l"].append(unlearned_rouge_l)
                    
                    case_result["probes"].append({
                        "type": "retain",
                        "question": probe_data['question'],
                        "expected_answer": probe_data['answer'],
                        "clean_model_answer": clean_answers.get("Retain", {}).get("text"),
                        "unlearned_model_answer": unlearned_answer,
                        "evaluation": {
                            "did_retain": did_retain,
                            "clean_prob": clean_answers.get("Retain", {}).get("prob"),
                            "unlearned_prob": unlearned_prob,
                            "clean_rouge_l": clean_answers.get("Retain", {}).get("rouge-l"),
                            "unlearned_rouge_l": unlearned_rouge_l
                        }
                    })
                    logger.info(f"🔹 Probe: Retain -> {'✅ RETAINED' if did_retain else '❌ FORGOT'}")
                    logger.info(f"  Question: {probe_data['question']}")
                    logger.info(f"  Clean Model Answer: {clean_answers.get('Retain', {}).get('text')} (Prob: {clean_answers.get('Retain', {}).get('prob', 0.0):.4f}, ROUGE-L: {clean_answers.get('Retain', {}).get('rouge-l', 0.0):.4f})")
                    logger.info(f"  Unlearned Model Answer: {unlearned_answer} (Prob: {unlearned_prob:.4f}, ROUGE-L: {unlearned_rouge_l:.4f})")
                logger.info("\n" + "=" * 80)

            detailed_results.append(case_result)

        if os.path.exists(self.temp_model_state_path):
            os.remove(self.temp_model_state_path)
            logger.info("Cleaned up temporary model state.")

        # --- Save Results ---
        output_dir = self.eval_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_summary_path = os.path.join(output_dir, "ripple_unlearning_detailed_results.json")
        with open(detailed_summary_path, 'w') as f:
            json.dump(detailed_results, f, indent=4)
        logger.info(f"Detailed evaluation results saved to {detailed_summary_path}")

        # Calculate and save aggregated summary
        final_results = {}
        for key, values in aggregated_results.items():
            final_results[f"mean_{key}"] = sum(values) / len(values) if values else 0.0
        
        final_results["skipped_cases"] = skipped_cases
        final_results["total_cases"] = len(dataset)
        final_results["evaluated_cases"] = len(dataset) - skipped_cases
        logger.info(f"Final Aggregated Results: {final_results}")
        
        summary_path = os.path.join(output_dir, "ripple_unlearning_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        logger.info(f"Aggregated evaluation summary saved to {summary_path}")

        return final_results

