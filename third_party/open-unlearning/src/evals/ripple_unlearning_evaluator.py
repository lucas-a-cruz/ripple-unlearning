# ruff: noqa
import json
import logging
import os

# Fix for huggingface/tokenizers warning when using dataloader_num_workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import gc
import multiprocessing
import re
from collections import defaultdict
from typing import Any, Dict, List

import evaluate
import numpy as np
import torch
from data.collators import DataCollatorForSupervisedDataset
from data.ripple_dataset import RippleUnlearningDataset
from data.unlearn import ForgetRetainDataset
from evals.metrics import ripple_metrics
from evals.metrics.ripple_metrics import check_answers
from evals.metrics.utils import evaluate_probability
# Importação necessária para criar batches manuais
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from trainer import load_trainer
from transformers import (AutoModelForCausalLM, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)

from .base import Evaluator

logger = logging.getLogger(__name__)

class RippleEvalCallback(TrainerCallback):
    """
    Custom Callback to evaluate the model on specific probes at the end of each epoch.
    """
    def __init__(self, evaluator, probes_to_log, perturbed_answers_map, rouge_scorer, history_list):
        self.evaluator = evaluator
        self.probes_to_log = probes_to_log
        self.perturbed_answers_map = perturbed_answers_map
        self.rouge_scorer = rouge_scorer
        self.history_list = history_list

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']
        
        # Switch to eval mode to disable dropout/etc for consistent metrics
        was_training = model.training
        model.eval()
        
        epoch_metrics = {
            "epoch": state.epoch,
            "step": state.global_step,
            "probes": {}
        }
        
        logger.info(f"\n--- ⏱️ Epoch {state.epoch} Evaluation ---")
        
        with torch.no_grad():
            for probe_name, probe_data in self.probes_to_log.items():
                if probe_data:
                    # Reuse the centralized evaluation logic
                    metrics = self.evaluator._evaluate_single_probe(
                        model, 
                        probe_name, 
                        probe_data, 
                        self.perturbed_answers_map.get(probe_name),
                        self.rouge_scorer
                    )
                    epoch_metrics["probes"][probe_name] = metrics
                    
                    # Log brief status
                    tr_str = f"{metrics.get('truth_ratio', 0.0):.4f}"
                    logger.info(f"  {probe_name}: TR={tr_str} | Prob={metrics['prob']:.4f} | ROUGE={metrics['rouge_l']:.4f}")

        self.history_list.append(epoch_metrics)
        
        # Restore training state
        if was_training:
            model.train()

class RippleUnlearningEvaluator(Evaluator):
    """
    Custom evaluator for the Ripple Unlearning Benchmark.
    This evaluator performs a one-by-one evaluation of unlearning cases.
    """
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("ripple_unlearning", eval_cfg=eval_cfg, **kwargs)
        self.trainer = None
        self.temp_model_state_path = "temp_model_state.pt"
        
        # OTIMIZAÇÃO: Detectar e configurar uso de CPU
        self.num_cores = multiprocessing.cpu_count()
        # Reserva 1 core para o sistema/gerenciamento se tivermos muitos, senão usa todos
        self.worker_threads = max(1, self.num_cores - 1) if self.num_cores > 4 else self.num_cores
        
        logger.info(f"🚀 Optimizing for {self.num_cores} CPU cores. Using {self.worker_threads} worker threads.")
        torch.set_num_threads(self.num_cores)

    def _compute_batch_metrics(self, model, question: str, answers: List[str]) -> List[Dict[str, float]]:
        """
        Calculates loss and probability for a list of answers in a SINGLE BATCH (Parallelized on GPU).
        This replaces the slow sequential loop.
        """
        if not answers:
            return []
            
        # 1. Tokenize all (question, answer) pairs
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        for ans in answers:
            # Note: _tokenize_qa returns tensors, we need to handle them carefully
            item = self._tokenize_qa(self.tokenizer, self.template_args, question, ans)
            input_ids_list.append(item["input_ids"])
            labels_list.append(item["labels"])
            attention_mask_list.append(item["attention_mask"])
            
        # 2. Pad sequence to create a batch
        # Ensure we have a pad token
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        # Collate (Pad inputs to max length in this batch)
        input_ids_batch = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id).to(model.device)
        labels_batch = pad_sequence(labels_list, batch_first=True, padding_value=-100).to(model.device)
        attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(model.device)
        
        # 3. Batch Forward Pass (The actual optimization)
        # We calculate loss manually here to handle the batch correctly
        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits
        
        # 4. Calculate Loss per item
        # Shift logits and labels for CausalLM loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_batch[..., 1:].contiguous()
        
        # Use reduction='none' to get loss per token, then we sum per row
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        # Flatten for API compatibility
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # Reshape back to [batch, seq_len]
        token_losses = token_losses.view(shift_labels.size())
        
        # Sum loss over the sequence (masked positions are 0.0)
        sum_losses = token_losses.sum(dim=1)
        # Count non-masked tokens to compute average
        non_masked_tokens = (shift_labels != -100).sum(dim=1)
        
        # Avoid division by zero
        avg_losses = sum_losses / (non_masked_tokens + 1e-9)
        
        results = []
        for loss_val in avg_losses:
            l = loss_val.item()
            # Prob = exp(-loss)
            results.append({'loss': l, 'prob': np.exp(-l)})
            
        return results

    def _get_loss_and_prob_for_answers(self, model, question, answers) -> Dict[str, float]:
        """
        Calculates the minimum loss and corresponding max probability over a list of possible answers.
        Optimized to use batch processing.
        """
        if not answers:
            return {'loss': float('inf'), 'prob': 0.0}
        
        if isinstance(answers, str):
            answers = [answers]

        # Use the batched computation
        batch_results = self._compute_batch_metrics(model, question, answers)
        
        if not batch_results:
             return {'loss': float('inf'), 'prob': 0.0}

        # Find the best result (min loss)
        best_result = min(batch_results, key=lambda x: x['loss'])
        
        return best_result

    def _get_perturbed_answers(self, question: str, answer: str) -> List[str]:
        """
        Generates plausible but incorrect answers for a given question-answer pair
        by prompting the model itself.
        """
        # Few-shot prompting works best for smaller models to enforce format and conciseness.
        prompt = (
            "Task: Provide one plausible but incorrect answer for the following question. "
            "The answer should be of the same entity type as the correct answer but factually wrong. "
            "Keep the answer very short (1-5 words).\n\n"
            "Question: What is the capital of France?\n"
            "Correct Answer: Paris\n"
            "Incorrect Answer: London\n\n"
            "Question: Who wrote Harry Potter?\n"
            "Correct Answer: J.K. Rowling\n"
            "Incorrect Answer: Stephen King\n\n"
            "Question: What is the currency of Japan?\n"
            "Correct Answer: Yen\n"
            "Incorrect Answer: Won\n\n"
            f"Question: {question}?\n"
            f"Correct Answer: {answer}\n"
            "Incorrect Answer:"
        )
        
        dummy_probe = {"question": prompt, "answer": ""} 
        response_text = self._get_answer_for_probe(self.model, self.tokenizer, self.template_args, dummy_probe)
        
        # Clean up the response
        perturbed_answer = response_text.strip().split('\n')[0].strip()
        
        # Remove any potential surrounding quotes if the model added them
        if perturbed_answer.startswith('"') and perturbed_answer.endswith('"'):
            perturbed_answer = perturbed_answer[1:-1]
            
        if perturbed_answer:
             return [perturbed_answer]
        
        logger.warning(f"Failed to generate perturbed answer for '{question}' (got empty response)")
        return []

    def _get_truth_ratio(self, model_to_eval, question: str, correct_answer: str, perturbed_answers: List[str]) -> float:
        """
        Calculates the truth ratio for a given question, correct answer, and a list of perturbed answers.
        truth_ratio = mean(prob_perturbed) / (prob_correct + 1e-10)
        
        OPTIMIZED: Uses batch processing to calculate all probabilities in one go.
        """
        if not perturbed_answers or not correct_answer:
            return 0.0
            
        # Combine correct answer and perturbed answers into a single batch request
        # This avoids looping and doing multiple small forward passes
        all_candidates = [correct_answer] + perturbed_answers
        
        # Get all metrics in one batch forward pass
        all_metrics = self._compute_batch_metrics(model_to_eval, question, all_candidates)
        
        if not all_metrics:
            return 0.0
            
        # Extract correct prob (first item) and perturbed probs (rest)
        prob_correct = all_metrics[0]['prob']
        
        # Note: perturbed_probs should be the mean of probabilities of perturbed answers
        perturbed_results = all_metrics[1:]
        perturbed_probs = [m['prob'] for m in perturbed_results]

        if prob_correct == 0.0: 
            return 0.0

        if not perturbed_probs:
            return 0.0

        avg_prob_perturbed = np.mean(perturbed_probs)
        return prob_correct / (prob_correct + avg_prob_perturbed + 1e-10)


    @staticmethod
    def _tokenize_qa(tokenizer, template_args, question: str, answer: str) -> Dict[str, torch.Tensor]:
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
            
            item = {"input_ids": torch.tensor(tokenized_ids), "attention_mask": torch.tensor([1] * len(tokenized_ids)), "labels": torch.tensor(labels)}
        else: 
            full_text = f"{question} {answer}"
            tokenized = tokenizer(full_text, add_special_tokens=False)
            question_tokens = tokenizer(question, add_special_tokens=False)['input_ids']
            q_len = len(question_tokens)

            labels = list(tokenized['input_ids'])
            labels[:q_len] = [-100] * q_len
            
            item = {"input_ids": torch.tensor(tokenized['input_ids']), "attention_mask": torch.tensor(tokenized['attention_mask']), "labels": torch.tensor(labels)}
        
        return item

    @staticmethod
    def _get_answer_for_probe(model, tokenizer, template_args, probe, log_prompt: bool = False) -> str:
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
        else: 
            if log_prompt: logger.info(f"\n[PROMPT FED TO MODEL]:\n---\n{question}\n---")
            inputs = tokenizer(question, return_tensors="pt").to(model.device)
            input_length = inputs['input_ids'].shape[1]

        stop_ids = [tokenizer.eos_token_id] + tokenizer.encode("\n", add_special_tokens=False)
        
        # Ensure generation config is safe
        outputs = model.generate(
            **inputs, 
            max_new_tokens=25, 
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=stop_ids,
            do_sample=False # Deterministic for eval
        )
        return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    def _evaluate_single_probe(self, model, probe_name, probe_data, perturbed_answers, rouge_scorer):
        """
        Helper method to evaluate a single probe. 
        Returns a dictionary of metrics.
        """
        text_answer = self._get_answer_for_probe(model, self.tokenizer, self.template_args, probe_data)
        metrics = self._get_loss_and_prob_for_answers(model, probe_data["question"], probe_data.get("answer"))
        
        ref_answer = probe_data.get("answer")
        if isinstance(ref_answer, list) and ref_answer:
            ref_answer = ref_answer[0]
        elif not isinstance(ref_answer, str):
            ref_answer = ""
        
        rouge_result = rouge_scorer.compute(predictions=[text_answer], references=[ref_answer], use_stemmer=True)
        
        # Calculate TRUTH RATIO
        tr_value = 0.0
        if perturbed_answers:
            tr_value = self._get_truth_ratio(model, probe_data["question"], probe_data.get("answer"), perturbed_answers)

        result = {
            "text": text_answer, 
            "loss": metrics['loss'], 
            "prob": metrics['prob'], 
            "rouge_l": rouge_result['rougeL'],
            "truth_ratio": tr_value
        }
        
        # Add simple boolean flags for convenience
        has_answer = check_answers(text_answer, probe_data.get("answer", ""))
        if probe_name == "Forget":
            result["did_forget"] = not has_answer
        elif probe_name == "Consistency":
            result["is_consistent"] = not has_answer
        elif probe_name == "Retain":
            result["did_retain"] = has_answer
            
        return result

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

        # WARNING: This hardcoded limit was in the original code. 
        # Uncomment the line below if you want to test only the first 25 cases.
        # dataset.data = dataset.data[:25]
        # logger.info(f"🔪 Limiting evaluation to the first 25 cases for testing.")

        # Save initial state to reset after each case
        torch.save(model.state_dict(), self.temp_model_state_path)
        
        aggregated_results, detailed_results = defaultdict(list), []
        skipped_cases = 0

        # Create output directory early to allow incremental saving
        output_dir = self.eval_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        detailed_json_path = os.path.join(output_dir, "ripple_unlearning_detailed_results.json")

        for case in tqdm(dataset, desc="Evaluating Ripple Unlearning Cases"):
            try:
                # Initialize history for this case
                case_history = []
                
                case_result = {"case_id": case.get("case_id"), "metadata": case.get("metadata"), "passed_pre_check": True, "probes": []}

                probes_to_log = {
                    "Forget": case.get("forget_probes", [])[0] if case.get("forget_probes") else None,
                    "Consistency": case.get("consistency_probes", [])[0] if case.get("consistency_probes") else None,
                    "Retain": case.get("retain_probes", [])[0] if case.get("retain_probes") else None,
                }
                
                clean_answers = {}
                perturbed_answers_map = {}

                # --- PRE-GENERATION OF PERTURBED ANSWERS FOR ALL PROBES ---
                for probe_name, probe_data in probes_to_log.items():
                    if probe_data:
                        p_answers = self._get_perturbed_answers(probe_data["question"], probe_data.get("answer"))
                        perturbed_answers_map[probe_name] = p_answers
                        
                        if p_answers:
                            logger.info(f"Generated perturbed answer for [{probe_name}] probe (Case {case.get('case_id')}): {p_answers[0]}")
                        else:
                            logger.warning(f"⚠️ Failed to generate perturbed answers for [{probe_name}] probe.")

                # --- CLEAN EVALUATION (Epoch 0) ---
                epoch_0_metrics = {"epoch": 0, "probes": {}}
                logger.info(f"\n--- ⏱️ Epoch 0 (Clean Model) Evaluation ---")
                for name, probe in probes_to_log.items():
                    if probe:
                        metrics = self._evaluate_single_probe(model, name, probe, perturbed_answers_map.get(name), rouge_scorer)
                        clean_answers[name] = metrics
                        epoch_0_metrics["probes"][name] = metrics

                        # Log brief status
                        tr_str = f"{metrics.get('truth_ratio', 0.0):.4f}"
                        logger.info(f"  {name}: TR={tr_str} | Prob={metrics['prob']:.4f} | ROUGE={metrics['rouge_l']:.4f}")
                
                case_history.append(epoch_0_metrics)
                        
                # --- UNLEARNING STEP ---
                # Load CLEAN state before training for this specific case
                model.load_state_dict(torch.load(self.temp_model_state_path))
                
                trainer_cfg = self.eval_cfg.get("trainer")
                # Ensure we don't leak memory with previous trainer instances
                if self.trainer:
                    del self.trainer
                    torch.cuda.empty_cache()

                self.trainer, trainer_args = load_trainer(trainer_cfg, model=model, train_dataset=[], data_collator=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer))
                
                # OTIMIZAÇÃO: Usar múltiplos workers para carregar dados
                trainer_args.dataloader_num_workers = self.worker_threads
                # Desabilita o pin_memory se usar muitos workers em máquinas com pouca RAM, mas geralmente ajuda na GPU
                trainer_args.dataloader_pin_memory = True 
                
                trainer_args.remove_unused_columns = False
                self.trainer.args = trainer_args
                
                forget_request = case["forget_request"]
                forget_dataset = [self._tokenize_qa(self.tokenizer, self.template_args, forget_request["question"], forget_request["answer"])]
                
                retain_probes = case.get("retain_probes", [])
                retain_answers_list_of_lists = [p.get("answer") for p in retain_probes if p.get("answer")]
                retain_first_answers = [ans[0] for ans in retain_answers_list_of_lists if ans]

                self.trainer.train_dataset = ForgetRetainDataset(
                    forget=forget_dataset,
                    retain=[self._tokenize_qa(self.tokenizer, self.template_args, p["question"], ans) for p, ans in zip(retain_probes, retain_first_answers)] or [self._tokenize_qa(self.tokenizer, self.template_args, " ", " ")]
                )

                # REGISTER CALLBACK
                self.trainer.add_callback(RippleEvalCallback(self, probes_to_log, perturbed_answers_map, rouge_scorer, case_history))

                self.trainer.train()
                
                # --- AGGREGATE RESULTS ---
                # We take the final state from case_history (last epoch)
                final_metrics = case_history[-1]["probes"]
                
                with torch.no_grad():
                    logger.info(f"\n{'='*80}\n📊 FINAL ANALYSIS FOR CASE: {case.get('case_id', 'N/A')}\n{'='*80}")
                    
                    for probe_type_name in ["Forget", "Consistency", "Retain"]:
                        if probe_type_name in final_metrics:
                            final_res = final_metrics[probe_type_name]
                            clean_res = clean_answers.get(probe_type_name, {})
                            
                            # Prepare simplified eval_dict for aggregation logic
                            eval_dict = {
                                "clean_truth_ratio": clean_res.get("truth_ratio", 0.0),
                                "unlearned_truth_ratio": final_res.get("truth_ratio", 0.0),
                                "clean_rouge_l": clean_res.get("rouge_l", 0.0),
                                "unlearned_rouge_l": final_res.get("rouge_l", 0.0),
                                # Pass through booleans
                                "did_forget": final_res.get("did_forget"),
                                "is_consistent": final_res.get("is_consistent"),
                                "did_retain": final_res.get("did_retain")
                            }

                            # Metrics aggregation logic
                            if probe_type_name == "Forget":
                                aggregated_results["forget_efficacy_rate"].append(1.0 if final_res["did_forget"] else 0.0)
                                log_status = f"Forget -> {'✅ FORGOT' if final_res['did_forget'] else '❌ FAILED TO FORGET'}"
                            elif probe_type_name == "Consistency":
                                aggregated_results["logical_inconsistency_rate"].append(0.0 if final_res["is_consistent"] else 1.0)
                                log_status = f"Consistency -> {'✅ CONSISTENT (Forgot)' if final_res['is_consistent'] else '❌ INCONSISTENT (Remembered)'}"
                            else: # Retain
                                aggregated_results["retain_accuracy_rate"].append(1.0 if final_res["did_retain"] else 0.0)
                                log_status = f"Retain -> {'✅ RETAINED' if final_res['did_retain'] else '❌ FORGOT'}"

                            # Aggregate numeric values
                            for key, val in eval_dict.items():
                                 if isinstance(val, (int, float)) and ('clean' in key or 'unlearned' in key):
                                    key_name = f"{key.replace('clean_','clean_'+probe_type_name.lower()+'_').replace('unlearned_','unlearned_'+probe_type_name.lower()+'_')}"
                                    aggregated_results[key_name].append(val)

                            # Add simplified structure to final output
                            case_result["probes"].append({
                                "type": probe_type_name.lower(),
                                "clean_model_answer": clean_res.get("text"),
                                "unlearned_model_answer": final_res.get("text"),
                                "evaluation": eval_dict
                            })
                            
                            logger.info(f"🔹 Probe: {log_status}")
                            logger.info(f"  Clean TR: {clean_res.get('truth_ratio', 0.0):.4f} -> Final TR: {final_res.get('truth_ratio', 0.0):.4f}")

                # ATTACH HISTORY TO RESULT
                case_result["history"] = case_history
                detailed_results.append(case_result)

                # SAVE INCREMENTALLY (Critical for long runs)
                with open(detailed_json_path, 'w') as f:
                    json.dump(detailed_results, f, indent=4)

            except Exception as e:
                logger.error(f"❌ Error evaluating case {case.get('case_id')}: {str(e)}")
                skipped_cases += 1
            
            # Cleanup memory after each case
            gc.collect()
            torch.cuda.empty_cache()

        if os.path.exists(self.temp_model_state_path): 
            os.remove(self.temp_model_state_path)
            logger.info("Cleaned up temporary model state.")
        
        logger.info(f"Detailed evaluation results saved to {detailed_json_path}")
        
        final_results = {f"mean_{key}": sum(values) / len(values) if values else 0.0 for key, values in aggregated_results.items()}
        final_results.update({"skipped_cases": skipped_cases, "total_cases": len(dataset), "evaluated_cases": len(dataset) - skipped_cases})
        logger.info(f"Final Aggregated Results: {final_results}")
        
        with open(os.path.join(output_dir, "ripple_unlearning_summary.json"), 'w') as f: 
            json.dump(final_results, f, indent=4)
        logger.info(f"Aggregated evaluation summary saved to {os.path.join(output_dir, 'ripple_unlearning_summary.json')}")

        return final_results