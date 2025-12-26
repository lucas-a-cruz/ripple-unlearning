# ruff: noqa
import json
import logging
import os
import time  # Added for sleep/wait logic

# Fix for huggingface/tokenizers warning when using dataloader_num_workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from torch.nn.utils.rnn import pad_sequence  # Import necessário para Batching
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
        # CONFIGURAÇÃO DE CHECKPOINTS:
        # Avalia nas epochs 3, 6 e 10.
        target_epochs = {3, 6, 10}
        current_epoch = int(round(state.epoch))
        
        is_target = current_epoch in target_epochs
        is_last = state.epoch >= args.num_train_epochs - 0.1
        
        if not (is_target or is_last):
            return

        model = kwargs['model']
        
        # Switch to eval mode
        was_training = model.training
        model.eval()
        
        epoch_metrics = {
            "epoch": state.epoch,
            "step": state.global_step,
            "probes": {}
        }
        
        logger.info(f"\n--- ⏱️ Evaluation Checkpoint (Epoch {state.epoch:.1f}) ---")
        
        try:
            with torch.no_grad():
                for probe_name, probe_data in self.probes_to_log.items():
                    if probe_data:
                        metrics = self.evaluator._evaluate_single_probe(
                            model, 
                            probe_name, 
                            probe_data, 
                            self.perturbed_answers_map.get(probe_name),
                            self.rouge_scorer
                        )
                        epoch_metrics["probes"][probe_name] = metrics
                        
                        tr_str = f"{metrics.get('truth_ratio', 0.0):.4f}"
                        logger.info(f"  {probe_name}: TR={tr_str} | Prob={metrics['prob']:.4f} | ROUGE={metrics['rouge_l']:.4f}")
        except Exception as e:
            logger.error(f"Error during callback evaluation: {e}")
        finally:
            self.history_list.append(epoch_metrics)
            # Restore training state
            if was_training:
                model.train()

class RippleUnlearningEvaluator(Evaluator):
    """
    Custom evaluator for the Ripple Unlearning Benchmark.
    """
    def __init__(self, eval_cfg, silent: bool = False, **kwargs):
        super().__init__("ripple_unlearning", eval_cfg=eval_cfg, **kwargs)
        self.trainer = None
        self.temp_model_state_path = "temp_model_state.pt"
        
        if silent:
            logger.setLevel(logging.WARNING)
        
        # Usar 0 workers para evitar overhead de multiprocessamento em datasets pequenos
        self.worker_threads = 0
        logger.info("🚀 Stability Fix: Using 0 dataloader workers to prevent resource exhaustion.")

    def _wait_for_memory(self, min_free_gib=1.5, timeout=120):
        """
        Bloqueia a execução se a memória livre da GPU for muito baixa.
        Isso evita que o script tente alocar memória quando a GPU já está cheia.
        """
        if not torch.cuda.is_available():
            return

        start_time = time.time()
        first_wait = True
        
        while True:
            # Obtém info da GPU 0 (assumindo single-gpu ou device correto setado por env)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gib = free_bytes / (1024**3)
            except Exception:
                # Se falhar ao pegar info, assume que está ok para tentar
                return

            if free_gib >= min_free_gib:
                if not first_wait:
                    logger.info(f"✅ Memory freed! {free_gib:.2f} GiB available. Resuming...")
                return

            # Se estamos aqui, a memória está cheia
            if first_wait:
                logger.warning(f"⏳ Low GPU Memory ({free_gib:.2f} GiB free). Waiting for release (Target: {min_free_gib} GiB)...")
                first_wait = False

            # Tenta limpar o que pode
            torch.cuda.empty_cache()
            gc.collect()
            
            if time.time() - start_time > timeout:
                logger.error(f"⚠️ Timeout waiting for memory after {timeout}s. Proceeding at risk...")
                return
            
            time.sleep(5)  # Espera 5 segundos antes de checar de novo

    def _compute_batch_metrics(self, model, question: str, answers: List[str]) -> List[Dict[str, float]]:
        """
        Calculates metrics for multiple answers in parallel using GPU batching.
        Includes chunking to prevent OOM on large batches.
        """
        if not answers:
            return []

        # MEMORY SAFEGUARD: Reduced chunk size to prevent OOM spikes
        chunk_size = 4
        all_results = []

        for i in range(0, len(answers), chunk_size):
            chunk_answers = answers[i : i + chunk_size]
            
            # 1. Tokenize chunk
            input_ids_list = []
            labels_list = []
            attention_mask_list = []

            for ans in chunk_answers:
                item = self._tokenize_qa(self.tokenizer, self.template_args, question, ans)
                if item['input_ids'].dim() > 1:
                    item['input_ids'] = item['input_ids'].squeeze(0)
                    item['labels'] = item['labels'].squeeze(0)
                    item['attention_mask'] = item['attention_mask'].squeeze(0)
                
                input_ids_list.append(item["input_ids"])
                labels_list.append(item["labels"])
                attention_mask_list.append(item["attention_mask"])

            # 2. Pad batch
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id
            
            input_ids_batch = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id).to(model.device)
            labels_batch = pad_sequence(labels_list, batch_first=True, padding_value=-100).to(model.device)
            attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(model.device)

            # 3. Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                logits = outputs.logits

                # 4. Calculate Loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels_batch[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size())
                
                loss_sum = loss.sum(dim=1)
                valid_tokens = (shift_labels != -100).sum(dim=1)
                avg_loss = loss_sum / (valid_tokens + 1e-9)
                
                for idx in range(len(chunk_answers)):
                    l = avg_loss[idx].item()
                    all_results.append({'loss': l, 'prob': np.exp(-l)})
                
                # Limpeza explícita de tensores intermediários grandes
                del outputs, logits, shift_logits, shift_labels, loss, input_ids_batch, labels_batch, attention_mask_batch
                torch.cuda.empty_cache() # Limpa cache após cada chunk pesado
        
        return all_results

    def _get_loss_and_prob_for_answers(self, model, question, answers) -> Dict[str, float]:
        if not answers:
            return {'loss': float('inf'), 'prob': 0.0}
        
        if isinstance(answers, str):
            answers = [answers]

        batch_results = self._compute_batch_metrics(model, question, answers)
        if not batch_results:
            return {'loss': float('inf'), 'prob': 0.0}

        return min(batch_results, key=lambda x: x['loss'])

    def _get_perturbed_answers(self, question: str, answer: str) -> List[str]:
        prompt = (
            f"Question: {question}?\n"
            f"Correct Answer: {answer}\n"
            "Task: Provide one short, plausible but factually incorrect answer.\n"
            "Incorrect Answer:"
        )
        
        dummy_probe = {"question": prompt, "answer": ""} 
        response_text = self._get_answer_for_probe(self.model, self.tokenizer, self.template_args, dummy_probe)
        perturbed_answer = response_text.strip().split('\n')[0].strip()
        
        if perturbed_answer.startswith('"') and perturbed_answer.endswith('"'):
            perturbed_answer = perturbed_answer[1:-1]
            
        if perturbed_answer:
             return [perturbed_answer]
        return []

    def _get_truth_ratio(self, model_to_eval, question: str, correct_answer: str, perturbed_answers: List[str]) -> float:
        if not perturbed_answers or not correct_answer:
            return 0.0
        
        all_candidates = [correct_answer] + perturbed_answers
        all_metrics = self._compute_batch_metrics(model_to_eval, question, all_candidates)
        
        if not all_metrics: return 0.0
            
        prob_correct = all_metrics[0]['prob']
        perturbed_probs = [m['prob'] for m in all_metrics[1:]]

        if prob_correct == 0.0 or not perturbed_probs:
            return 0.0

        avg_prob_perturbed = np.mean(perturbed_probs)
        return prob_correct / (prob_correct + avg_prob_perturbed + 1e-10)

    @staticmethod
    def _tokenize_qa(tokenizer, template_args, question: str, answer: str) -> Dict[str, torch.Tensor]:
        chat = []
        if template_args.get("apply_chat_template"):
            if sys_p := template_args.get("system_prompt"):
                chat.append({"role": "system", "content": sys_p})
            chat.append({"role": "user", "content": question})
            chat.append({"role": "assistant", "content": answer})
            tokenized_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
        else:
            tokenized_ids = tokenizer(f"{question} {answer}", add_special_tokens=False)['input_ids']
            
        labels = list(tokenized_ids)
        return {"input_ids": torch.tensor(tokenized_ids), "attention_mask": torch.tensor([1] * len(tokenized_ids)), "labels": torch.tensor(labels)}

    @staticmethod
    def _get_answer_for_probe(model, tokenizer, template_args, probe, log_prompt: bool = False) -> str:
        if not probe or "question" not in probe: return "Invalid Probe"
        question = probe["question"]
        
        chat = []
        if template_args.get("apply_chat_template"):
            if sys_p := template_args.get("system_prompt"):
                chat.append({"role": "system", "content": sys_p})
            chat.append({"role": "user", "content": question})
            inputs = {'input_ids': tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(model.device)}
            input_length = inputs['input_ids'].shape[1]
        else: 
            inputs = tokenizer(question, return_tensors="pt").to(model.device)
            input_length = inputs['input_ids'].shape[1]

        stop_ids = [tokenizer.eos_token_id] + tokenizer.encode("\n", add_special_tokens=False)
        
        # Greedy decoding for determinism and speed
        outputs = model.generate(
            **inputs, 
            max_new_tokens=25, 
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=stop_ids,
            do_sample=False
        )
        decoded_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        # Aggressive cleanup of generation tensors
        del inputs, outputs, stop_ids
        torch.cuda.empty_cache()
        
        return decoded_text

    def _evaluate_single_probe(self, model, probe_name, probe_data, perturbed_answers, rouge_scorer):
        text_answer = self._get_answer_for_probe(model, self.tokenizer, self.template_args, probe_data)
        metrics = self._get_loss_and_prob_for_answers(model, probe_data["question"], probe_data.get("answer"))
        
        ref = probe_data.get("answer", "")
        if isinstance(ref, list) and ref: ref = ref[0]
        
        rouge = rouge_scorer.compute(predictions=[text_answer], references=[ref], use_stemmer=True)
        tr = self._get_truth_ratio(model, probe_data["question"], probe_data.get("answer"), perturbed_answers) if perturbed_answers else 0.0

        return {
            "text": text_answer, 
            "loss": metrics['loss'], 
            "prob": metrics['prob'], 
            "rouge_l": rouge['rougeL'],
            "truth_ratio": tr,
            "did_forget": not check_answers(text_answer, ref) if probe_name == "Forget" else None,
            "is_consistent": not check_answers(text_answer, ref) if probe_name == "Consistency" else None,
            "did_retain": check_answers(text_answer, ref) if probe_name == "Retain" else None
        }

    def evaluate(self, model: AutoModelForCausalLM, **kwargs):
        self.model, self.tokenizer, self.template_args = model, kwargs.get("tokenizer"), kwargs.get("template_args")
        rouge_scorer = evaluate.load('rouge')
        dataset = RippleUnlearningDataset(path=self.eval_cfg.data.ripple_unlearning.args.path)
        
        torch.save(model.state_dict(), self.temp_model_state_path)
        
        aggregated_results, detailed_results = defaultdict(list), []
        skipped_cases = 0
        output_dir = self.eval_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use enumerate to track iterations for periodic cleanup
        for case_idx, case in enumerate(tqdm(dataset, desc="Evaluating Ripple Unlearning")):
            try:
                # 0. Wait for Memory (Throttle)
                # Ensure we have at least 1.0 GB free before starting a case
                self._wait_for_memory(min_free_gib=1.0)
                
                # 0.5 Periodic Deep Cleanup (Every 10 cases)
                if case_idx > 0 and case_idx % 10 == 0:
                    logger.info("🧹 Periodic Aggressive Cleanup (every 10 cases)...")
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(2) # Allow OS to reclaim
                    
                
                # 1. Clean Memory Aggressively from previous iter
                model.zero_grad(set_to_none=True)
                if self.trainer:
                    if hasattr(self.trainer, 'optimizer'): del self.trainer.optimizer
                    if hasattr(self.trainer, 'lr_scheduler'): del self.trainer.lr_scheduler
                    if hasattr(self.trainer, 'model_wrapped'): del self.trainer.model_wrapped
                    if hasattr(self.trainer, 'model'): del self.trainer.model
                    del self.trainer
                    self.trainer = None
                
                gc.collect()
                torch.cuda.empty_cache()
                
                # 2. Setup Case
                case_history = []
                case_id = case.get("case_id", "unknown")
                
                probes_to_log = {
                    "Forget": case.get("forget_probes", [])[0] if case.get("forget_probes") else None,
                    "Consistency": case.get("consistency_probes", [])[0] if case.get("consistency_probes") else None,
                    "Retain": case.get("retain_probes", [])[0] if case.get("retain_probes") else None,
                }
                
                # 3. Pre-compute Perturbed Answers (Clean Model)
                model.load_state_dict(torch.load(self.temp_model_state_path, map_location='cpu'))
                model.eval()
                
                perturbed_map = {}
                for pname, pdata in probes_to_log.items():
                    if pdata:
                        perturbed_map[pname] = self._get_perturbed_answers(pdata["question"], pdata.get("answer"))

                # 4. Epoch 0 Evaluation
                epoch_0_metrics = {"epoch": 0, "probes": {}}
                logger.info(f"\n--- Epoch 0 (Clean) Case {case_id} ---")
                with torch.no_grad():
                    for pname, pdata in probes_to_log.items():
                        if pdata:
                            metrics = self._evaluate_single_probe(model, pname, pdata, perturbed_map.get(pname), rouge_scorer)
                            epoch_0_metrics["probes"][pname] = metrics
                            logger.info(f"  {pname}: TR={metrics.get('truth_ratio', 0):.4f}")
                case_history.append(epoch_0_metrics)
                
                # 5. Training
                model.zero_grad(set_to_none=True)
                
                trainer_cfg = self.eval_cfg.get("trainer")
                self.trainer, trainer_args = load_trainer(trainer_cfg, model=model, train_dataset=[], data_collator=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer))
                
                trainer_args.dataloader_num_workers = 0 
                trainer_args.remove_unused_columns = False
                self.trainer.args = trainer_args
                
                forget_req = case["forget_request"]
                forget_ds = [self._tokenize_qa(self.tokenizer, self.template_args, forget_req["question"], forget_req["answer"])]
                
                retain_probes = case.get("retain_probes", [])
                retain_ans = [p.get("answer")[0] for p in retain_probes if p.get("answer")]
                retain_ds = [self._tokenize_qa(self.tokenizer, self.template_args, p["question"], a) for p, a in zip(retain_probes, retain_ans)] or [self._tokenize_qa(self.tokenizer, self.template_args, " ", " ")]

                self.trainer.train_dataset = ForgetRetainDataset(forget=forget_ds, retain=retain_ds)
                
                eval_cb = RippleEvalCallback(self, probes_to_log, perturbed_map, rouge_scorer, case_history)
                self.trainer.add_callback(eval_cb)

                self.trainer.train()
                
                # 6. Post-Case Cleanup
                self.trainer.remove_callback(eval_cb)
                del eval_cb
                if hasattr(self.trainer, 'optimizer'): del self.trainer.optimizer
                if hasattr(self.trainer, 'lr_scheduler'): del self.trainer.lr_scheduler
                del self.trainer
                self.trainer = None
                
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                
                # --- Result Aggregation ---
                final_metrics = case_history[-1]["probes"]
                clean_answers = {k: v for k,v in epoch_0_metrics["probes"].items()}
                
                case_res = {"case_id": case_id, "probes": [], "history": case_history}
                
                for ptype in ["Forget", "Consistency", "Retain"]:
                    if ptype in final_metrics:
                        fin, cln = final_metrics[ptype], clean_answers.get(ptype, {})
                        
                        eval_dict = {
                            "clean_truth_ratio": cln.get("truth_ratio", 0.0),
                            "unlearned_truth_ratio": fin.get("truth_ratio", 0.0),
                            "clean_rouge_l": cln.get("rouge_l", 0.0),
                            "unlearned_rouge_l": fin.get("rouge_l", 0.0),
                            "did_forget": fin.get("did_forget"),
                            "is_consistent": fin.get("is_consistent"),
                            "did_retain": fin.get("did_retain")
                        }
                        
                        if ptype == "Forget": aggregated_results["forget_efficacy_rate"].append(1.0 if fin["did_forget"] else 0.0)
                        elif ptype == "Consistency": aggregated_results["logical_inconsistency_rate"].append(0.0 if fin["is_consistent"] else 1.0)
                        else: aggregated_results["retain_accuracy_rate"].append(1.0 if fin["did_retain"] else 0.0)
                        
                        for k, v in eval_dict.items():
                            if isinstance(v, (int, float)) and ('clean' in k or 'unlearned' in k):
                                key_name = f"{k.replace('clean_',f'clean_{ptype.lower()}_').replace('unlearned_',f'unlearned_{ptype.lower()}_')}"
                                aggregated_results[key_name].append(v)
                                
                        case_res["probes"].append({"type": ptype.lower(), "evaluation": eval_dict})

                detailed_results.append(case_res)
                with open(os.path.join(output_dir, "ripple_unlearning_detailed_results.json"), 'w') as f:
                    json.dump(detailed_results, f, indent=4)

            except Exception as e:
                logger.error(f"❌ Error evaluating case {case.get('case_id', 'unknown')}: {str(e)}")
                skipped_cases += 1
                if self.trainer:
                    del self.trainer
                    self.trainer = None
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()

        if os.path.exists(self.temp_model_state_path): os.remove(self.temp_model_state_path)
        
        final_results = {f"mean_{k}": sum(v)/len(v) if v else 0.0 for k,v in aggregated_results.items()}
        with open(os.path.join(output_dir, "ripple_unlearning_summary.json"), 'w') as f: json.dump(final_results, f, indent=4)
        
        return final_results