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

    def _get_metrics_for_answers(self, model, question, answers) -> dict:
        """
        Calculates the minimum loss and corresponding max probability over a list of possible answers.
        """
        if not answers:
            return {'loss': float('inf'), 'prob': 0.0}
        
        # Ensure answers is a list, even if a single string is passed
        if isinstance(answers, str):
            answers = [answers]

        min_loss = float('inf')
        
        for answer in answers:
            if not answer or not isinstance(answer, str):
                continue

            tokenized_pair = self._tokenize_qa(self.tokenizer, self.template_args, question, answer)
            batch = {
                "input_ids": tokenized_pair['input_ids'].unsqueeze(0).to(model.device),
                "attention_mask": tokenized_pair['attention_mask'].unsqueeze(0).to(model.device),
                "labels": tokenized_pair['labels'].unsqueeze(0).to(model.device)
            }

            with torch.no_grad():
                prob_results = evaluate_probability(model, batch)
            
            if prob_results and prob_results[0] and "avg_loss" in prob_results[0]:
                current_loss = prob_results[0]["avg_loss"]
                if current_loss is not None:
                    min_loss = min(min_loss, current_loss)

        max_prob = torch.exp(-torch.tensor(min_loss)).item() if min_loss != float('inf') else 0.0
        return {'loss': min_loss if min_loss != float('inf') else -1.0, 'prob': max_prob}

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
            
            tokenized_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
            prompt_ids = tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True)
            q_len = len(prompt_ids)

            labels = list(tokenized_ids)
            labels[:q_len] = [-100] * q_len
            
            item = {"input_ids": tokenized_ids, "attention_mask": [1] * len(tokenized_ids), "labels": labels}
        else: # Fallback
            full_text = f"{question} {answer}"
            tokenized = tokenizer(full_text, add_special_tokens=False)
            question_tokens = tokenizer(question, add_special_tokens=False)['input_ids']
            q_len = len(question_tokens)

            labels = list(tokenized['input_ids'])
            labels[:q_len] = [-100] * q_len
            
            item = {"input_ids": tokenized['input_ids'], "attention_mask": tokenized['attention_mask'], "labels": labels}
        
        return {k: torch.tensor(v) for k, v in item.items()}

    @staticmethod
    def _get_answer_for_probe(model, tokenizer, template_args, probe, log_prompt: bool = False) -> str:
        """Generates a text answer for a given probe question."""
        if not probe or "question" not in probe: return "Invalid Probe"
        question = probe["question"]
        
        chat = []
        if template_args.get("apply_chat_template"):
            if system_prompt := template_args.get("system_prompt"):
                chat.append({"role": "system", "content": system_prompt})
            chat.append({"role": "user", "content": question})
            
            if log_prompt:
                prompt_for_logging = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                logger.info(f"\n[PROMPT FED TO MODEL]:\n---\n{prompt_for_logging}\n---")

            inputs = {'input_ids': tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(model.device)}
            input_length = inputs['input_ids'].shape[1]
        else: # Fallback
            if log_prompt: logger.info(f"\n[PROMPT FED TO MODEL]:\n---\n{question}\n---")
            inputs = tokenizer(question, return_tensors="pt").to(model.device)
            input_length = inputs['input_ids'].shape[1]

        stop_ids = [tokenizer.eos_token_id] + tokenizer.encode("\n", add_special_tokens=False)
        outputs = model.generate(**inputs, max_new_tokens=25, pad_token_id=tokenizer.eos_token_id, eos_token_id=stop_ids)
        return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    def evaluate(self, model: AutoModelForCausalLM, **kwargs):
        self.model, self.tokenizer, self.template_args = model, kwargs.get("tokenizer"), kwargs.get("template_args")
        if not all([self.tokenizer, self.template_args]):
            raise ValueError("Tokenizer and template_args must be provided.")

        rouge_scorer = evaluate.load('rouge')
        
        dataset_path = self.eval_cfg.data.ripple_unlearning.args.path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
        
        dataset = RippleUnlearningDataset(path=dataset_path)
        logger.info(f"Loaded Ripple Unlearning dataset with {len(dataset)} cases.")

        # Limit dataset to 25 samples for quick testing
        # dataset.data = dataset.data[:25]
        # logger.info(f"🔪 Limiting evaluation to the first 25 cases for testing.")

        torch.save(model.state_dict(), self.temp_model_state_path)
        
        aggregated_results, detailed_results = defaultdict(list), []
        skipped_cases = 0

        for case in tqdm(dataset, desc="Evaluating Ripple Unlearning Cases"):
            case_result = {"case_id": case.get("case_id"), "metadata": case.get("metadata"), "passed_pre_check": True, "probes": []}

            probes_to_log = {
                "Forget": case.get("forget_probes", [])[0] if case.get("forget_probes") else None,
                "Consistency": case.get("consistency_probes", [])[0] if case.get("consistency_probes") else None,
                "Retain": case.get("retain_probes", [])[0] if case.get("retain_probes") else None,
            }
            
            clean_answers = {}
            for name, probe in probes_to_log.items():
                if probe:
                    text_answer = self._get_answer_for_probe(model, self.tokenizer, self.template_args, probe)
                    metrics = self._get_metrics_for_answers(model, probe["question"], probe.get("answer"))
                    
                    ref_answer = probe.get("answer")
                    if isinstance(ref_answer, list) and ref_answer:
                        ref_answer = ref_answer[0]
                    elif not isinstance(ref_answer, str):
                        ref_answer = ""
                    
                    rouge_result = rouge_scorer.compute(predictions=[text_answer], references=[ref_answer], use_stemmer=True)
                    clean_answers[name] = {"text": text_answer, "loss": metrics['loss'], "prob": metrics['prob'], "rouge-l": rouge_result['rougeL']}

            model.load_state_dict(torch.load(self.temp_model_state_path))
            
            trainer_cfg = self.eval_cfg.get("trainer")
            self.trainer, trainer_args = load_trainer(trainer_cfg, model=model, train_dataset=[], data_collator=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer))
            logger.info(f"Loaded Unlearning Trainer: {self.trainer.__class__.__name__}")
            trainer_args.remove_unused_columns = False
            self.trainer.args = trainer_args
            
            retain_probes = case.get("retain_probes", [])
            retain_answers = [p.get("answer") for p in retain_probes if p.get("answer")]
            # Flatten the list of lists of answers and take the first answer of each
            retain_first_answers = [ans[0] for ans in retain_answers if ans]

            self.trainer.train_dataset = ForgetRetainDataset(
                forget=[self._tokenize_qa(self.tokenizer, self.template_args, case["forget_request"]["question"], case["forget_request"]["answer"])],
                retain=[self._tokenize_qa(self.tokenizer, self.template_args, p["question"], ans) for p, ans in zip(retain_probes, retain_first_answers)] or [self._tokenize_qa(self.tokenizer, self.template_args, " ", " ")]
            )

            self.trainer.train()
            
            with torch.no_grad():
                model.eval()
                logger.info(f"\n{'='*80}\n📊 ANALYSIS FOR CASE: {case.get('case_id', 'N/A')}\n{'='*80}")
                
                for probe_type_name, probes in [("Forget", case.get("forget_probes", [])), ("Consistency", case.get("consistency_probes", [])), ("Retain", case.get("retain_probes", []))]:
                    if probes:
                        probe_data = probes[0]
                        unlearned_answer = self._get_answer_for_probe(model, self.tokenizer, self.template_args, probe_data)
                        unlearned_metrics = self._get_metrics_for_answers(model, probe_data["question"], probe_data.get("answer"))
                        
                        ref_answer = probe_data.get("answer")
                        if isinstance(ref_answer, list) and ref_answer:
                            ref_answer = ref_answer[0]
                        elif not isinstance(ref_answer, str):
                            ref_answer = ""
                        
                        unlearned_rouge_result = rouge_scorer.compute(predictions=[unlearned_answer], references=[ref_answer], use_stemmer=True)
                        unlearned_rouge_l = unlearned_rouge_result['rougeL']

                        has_answer = check_answers(unlearned_answer, probe_data.get("answer", ""))
                        
                        eval_dict = {
                            "clean_loss": clean_answers.get(probe_type_name, {}).get("loss"), "unlearned_loss": unlearned_metrics['loss'],
                            "clean_prob": clean_answers.get(probe_type_name, {}).get("prob"), "unlearned_prob": unlearned_metrics['prob'],
                            "clean_rouge_l": clean_answers.get(probe_type_name, {}).get("rouge-l"), "unlearned_rouge_l": unlearned_rouge_l
                        }

                        if probe_type_name == "Forget":
                            did_forget = not has_answer
                            aggregated_results["forget_efficacy_rate"].append(1.0 if did_forget else 0.0)
                            eval_dict["did_forget"] = did_forget
                            log_status = f"Forget -> {'✅ FORGOT' if did_forget else '❌ FAILED TO FORGET'}"
                        elif probe_type_name == "Consistency":
                            is_consistent = not has_answer
                            aggregated_results["logical_inconsistency_rate"].append(0.0 if is_consistent else 1.0)
                            eval_dict["is_consistent"] = is_consistent
                            log_status = f"Consistency -> {'✅ CONSISTENT' if is_consistent else '❌ INCONSISTENT'}"
                        else: # Retain
                            did_retain = has_answer
                            aggregated_results["retain_accuracy_rate"].append(1.0 if did_retain else 0.0)
                            eval_dict["did_retain"] = did_retain
                            log_status = f"Retain -> {'✅ RETAINED' if did_retain else '❌ FORGOT'}"

                        for key, val in eval_dict.items():
                             if val is not None and ('clean' in key or 'unlearned' in key):
                                key_name = f"{key.replace('clean_','clean_'+probe_type_name.lower()+'_').replace('unlearned_','unlearned_'+probe_type_name.lower()+'_')}"
                                aggregated_results[key_name].append(val)

                        case_result["probes"].append({"type": probe_type_name.lower(), "question": probe_data['question'], "expected_answer": probe_data['answer'], "clean_model_answer": clean_answers.get(probe_type_name, {}).get("text"), "unlearned_model_answer": unlearned_answer, "evaluation": eval_dict})
                        
                        logger.info(f"🔹 Probe: {log_status}")
                        logger.info(f"  Question: {probe_data['question']}")
                        logger.info(f"  Clean Model Answer: {clean_answers.get(probe_type_name, {}).get('text')} (Loss: {eval_dict['clean_loss']:.4f}, Prob: {eval_dict['clean_prob']:.4f}, ROUGE-L: {eval_dict['clean_rouge_l']:.4f})")
                        logger.info(f"  Unlearned Model Answer: {unlearned_answer} (Loss: {eval_dict['unlearned_loss']:.4f}, Prob: {eval_dict['unlearned_prob']:.4f}, ROUGE-L: {eval_dict['unlearned_rouge_l']:.4f})")

                logger.info("\n" + "=" * 80)
            detailed_results.append(case_result)

        if os.path.exists(self.temp_model_state_path): os.remove(self.temp_model_state_path); logger.info("Cleaned up temporary model state.")
        output_dir = self.eval_cfg.output_dir; os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "ripple_unlearning_detailed_results.json"), 'w') as f: json.dump(detailed_results, f, indent=4)
        logger.info(f"Detailed evaluation results saved to {os.path.join(output_dir, 'ripple_unlearning_detailed_results.json')}")
        
        final_results = {f"mean_{key}": sum(values) / len(values) if values else 0.0 for key, values in aggregated_results.items()}
        final_results.update({"skipped_cases": skipped_cases, "total_cases": len(dataset), "evaluated_cases": len(dataset) - skipped_cases})
        logger.info(f"Final Aggregated Results: {final_results}")
        with open(os.path.join(output_dir, "ripple_unlearning_summary.json"), 'w') as f: json.dump(final_results, f, indent=4)
        logger.info(f"Aggregated evaluation summary saved to {os.path.join(output_dir, 'ripple_unlearning_summary.json')}")

        return final_results