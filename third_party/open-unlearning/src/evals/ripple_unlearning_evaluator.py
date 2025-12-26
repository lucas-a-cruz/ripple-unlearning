# ruff: noqa
import gc
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from typing import Any, Dict, List

import evaluate
import numpy as np
import torch
import torch.multiprocessing as mp
# Imports assumidos do seu ambiente
from data.collators import DataCollatorForSupervisedDataset
from data.ripple_dataset import RippleUnlearningDataset
from data.unlearn import ForgetRetainDataset
from evals.metrics.ripple_metrics import check_answers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from trainer import load_trainer
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments)

from .base import Evaluator

logger = logging.getLogger(__name__)

# --- FUNÇÃO DO TRABALHADOR (Executa em Processo Isolado) ---
def _run_case_in_worker(
    temp_dir: str,
    case_data: Dict,
    eval_cfg: Any,
    template_args: Dict,
    result_queue: mp.Queue
):
    """
    Carrega o modelo do zero, executa UM caso de teste, coloca o resultado na fila e encerra.
    O encerramento deste processo garante a limpeza forçada da VRAM pelo OS.
    """
    try:
        # 1. Setup Básico
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 2. Carregar Recursos (Do disco, garantindo isolamento)
        # Carrega Tokenizer e Config salvos pelo processo pai
        tokenizer = AutoTokenizer.from_pretrained(temp_dir)
        config = AutoConfig.from_pretrained(temp_dir)
        
        # Cria o modelo (vazio inicialmente para rapidez)
        model = AutoModelForCausalLM.from_config(config)
        
        # Carrega os pesos "limpos" salvos
        state_dict_path = os.path.join(temp_dir, "base_model.pt")
        # map_location='cpu' evita pico de memória na carga antes de mover para GPU
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # 3. Instancia uma versão leve do Evaluator apenas para usar os métodos auxiliares
        # (Não chamamos evaluate() dele, apenas usamos seus métodos estáticos/helpers)
        worker_eval = RippleUnlearningEvaluator(eval_cfg, silent=True)
        worker_eval.model = model
        worker_eval.tokenizer = tokenizer
        worker_eval.template_args = template_args

        # 4. Preparação do Caso (Lógica original)
        case_id = case_data.get("case_id", "unknown")
        probes_to_log = {
            "Forget": case_data.get("forget_probes", [])[0] if case_data.get("forget_probes") else None,
            "Consistency": case_data.get("consistency_probes", [])[0] if case_data.get("consistency_probes") else None,
            "Retain": case_data.get("retain_probes", [])[0] if case_data.get("retain_probes") else None,
        }

        # 5. Pre-compute Perturbed Answers (Clean Model)
        rouge_scorer = evaluate.load('rouge')
        perturbed_map = {}
        for pname, pdata in probes_to_log.items():
            if pdata:
                perturbed_map[pname] = worker_eval._get_perturbed_answers(pdata["question"], pdata.get("answer"))

        # 6. Epoch 0 Evaluation (Baseline)
        epoch_0_metrics = {"epoch": 0, "probes": {}}
        with torch.no_grad():
            for pname, pdata in probes_to_log.items():
                if pdata:
                    metrics = worker_eval._evaluate_single_probe(
                        model, pname, pdata, perturbed_map.get(pname), rouge_scorer
                    )
                    epoch_0_metrics["probes"][pname] = metrics

        case_history = [epoch_0_metrics]

        # 7. Training Phase
        model.train()
        
        # Preparar Datasets
        forget_req = case_data["forget_request"]
        forget_ds = [worker_eval._tokenize_qa(tokenizer, template_args, forget_req["question"], forget_req["answer"])]
        
        retain_probes = case_data.get("retain_probes", [])
        retain_ans = [p.get("answer")[0] for p in retain_probes if p.get("answer")]
        retain_ds = [worker_eval._tokenize_qa(tokenizer, template_args, p["question"], a) for p, a in zip(retain_probes, retain_ans)] or \
                    [worker_eval._tokenize_qa(tokenizer, template_args, " ", " ")]

        train_dataset = ForgetRetainDataset(forget=forget_ds, retain=retain_ds)

        # Carregar Trainer
        trainer_cfg = eval_cfg.get("trainer")
        trainer, trainer_args = load_trainer(
            trainer_cfg, 
            model=model, 
            train_dataset=train_dataset, 
            data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        )
        
        # Otimizações para o Worker
        trainer_args.dataloader_num_workers = 0 
        trainer_args.report_to = "none" # Desabilita wandb no worker para evitar conflito
        trainer_args.disable_tqdm = True # Limpa output
        
        # --- CORREÇÃO DO ERRO ---
        # Garante que o Trainer não remova colunas do dataset customizado
        # Se True (default), o Trainer remove 'input_ids' se não conseguir inferir a assinatura do modelo,
        # causando o erro "The batch received was empty".
        trainer_args.remove_unused_columns = False 
        
        # Opcional: Ativar BF16 se suportado (recomendado para A100)
        trainer_args.bf16 = True
        trainer_args.fp16 = False
        
        trainer.args = trainer_args

        # Callback
        eval_cb = RippleEvalCallback(worker_eval, probes_to_log, perturbed_map, rouge_scorer, case_history)
        trainer.add_callback(eval_cb)

        # Treinar
        trainer.train()

        # 8. Sucesso - Enviar dados
        result_queue.put({
            "status": "success", 
            "epoch_0_metrics": epoch_0_metrics,
            "history": case_history
        })

    except Exception as e:
        # Em caso de erro, loga e envia status de falha
        logger.error(f"Worker process failed for case {case_data.get('case_id')}: {e}")
        result_queue.put({"status": "error", "message": str(e)})
    finally:
        # Não precisamos de limpeza agressiva manual aqui (gc, empty_cache),
        # pois o processo vai morrer logo em seguida.
        pass


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
        target_epochs = {3, 6, 10}
        current_epoch = int(round(state.epoch))
        
        is_target = current_epoch in target_epochs
        is_last = state.epoch >= args.num_train_epochs - 0.1
        
        if not (is_target or is_last):
            return

        model = kwargs['model']
        was_training = model.training
        model.eval()
        
        epoch_metrics = {
            "epoch": state.epoch,
            "step": state.global_step,
            "probes": {}
        }
        
        # Log simplificado para não poluir console multiprocessado
        # logger.info(f"Worker Eval Epoch {state.epoch:.1f}")
        
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
        except Exception as e:
            logger.error(f"Error during callback evaluation: {e}")
        finally:
            self.history_list.append(epoch_metrics)
            if was_training:
                model.train()


class RippleUnlearningEvaluator(Evaluator):
    """
    Custom evaluator for the Ripple Unlearning Benchmark.
    Refactored to use Subprocess Isolation for robust memory management.
    """
    def __init__(self, eval_cfg, silent: bool = False, **kwargs):
        super().__init__("ripple_unlearning", eval_cfg=eval_cfg, **kwargs)
        if silent:
            logger.setLevel(logging.WARNING)
        
        # Configurar método de spawn para garantir contexto limpo de CUDA
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass # Já configurado

    def _compute_batch_metrics(self, model, question: str, answers: List[str]) -> List[Dict[str, float]]:
        # Same implementation, but WITHOUT empty_cache for speed
        if not answers: return []
        chunk_size = 4
        all_results = []

        for i in range(0, len(answers), chunk_size):
            chunk_answers = answers[i : i + chunk_size]
            
            input_ids_list, labels_list, attention_mask_list = [], [], []
            for ans in chunk_answers:
                item = self._tokenize_qa(self.tokenizer, self.template_args, question, ans)
                if item['input_ids'].dim() > 1:
                    item['input_ids'] = item['input_ids'].squeeze(0)
                    item['labels'] = item['labels'].squeeze(0)
                    item['attention_mask'] = item['attention_mask'].squeeze(0)
                input_ids_list.append(item["input_ids"])
                labels_list.append(item["labels"])
                attention_mask_list.append(item["attention_mask"])

            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            
            input_ids_batch = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id).to(model.device)
            labels_batch = pad_sequence(labels_list, batch_first=True, padding_value=-100).to(model.device)
            attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                logits = outputs.logits
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
                
                # Cleanup básico, mas o processo vai morrer de qualquer jeito
                del outputs, logits, shift_logits, shift_labels, loss, input_ids_batch
                # torch.cuda.empty_cache() REMOVIDO para performance dentro do chunk
        
        return all_results

    def _get_loss_and_prob_for_answers(self, model, question, answers) -> Dict[str, float]:
        if not answers: return {'loss': float('inf'), 'prob': 0.0}
        if isinstance(answers, str): answers = [answers]
        batch_results = self._compute_batch_metrics(model, question, answers)
        if not batch_results: return {'loss': float('inf'), 'prob': 0.0}
        return min(batch_results, key=lambda x: x['loss'])

    def _get_perturbed_answers(self, question: str, answer: str) -> List[str]:
        prompt = (f"Question: {question}?\nCorrect Answer: {answer}\n"
                  "Task: Provide one short, plausible but factually incorrect answer.\n"
                  "Incorrect Answer:")
        dummy_probe = {"question": prompt, "answer": ""} 
        response_text = self._get_answer_for_probe(self.model, self.tokenizer, self.template_args, dummy_probe)
        perturbed_answer = response_text.strip().split('\n')[0].strip()
        if perturbed_answer.startswith('"') and perturbed_answer.endswith('"'):
            perturbed_answer = perturbed_answer[1:-1]
        return [perturbed_answer] if perturbed_answer else []

    def _get_truth_ratio(self, model_to_eval, question: str, correct_answer: str, perturbed_answers: List[str]) -> float:
        if not perturbed_answers or not correct_answer: return 0.0
        all_candidates = [correct_answer] + perturbed_answers
        all_metrics = self._compute_batch_metrics(model_to_eval, question, all_candidates)
        if not all_metrics: return 0.0
        prob_correct = all_metrics[0]['prob']
        perturbed_probs = [m['prob'] for m in all_metrics[1:]]
        if prob_correct == 0.0 or not perturbed_probs: return 0.0
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
        outputs = model.generate(
            **inputs, max_new_tokens=25, pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=stop_ids, do_sample=False
        )
        decoded_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        del inputs, outputs, stop_ids
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
        # 1. SETUP INICIAL
        self.tokenizer = kwargs.get("tokenizer")
        self.template_args = kwargs.get("template_args")
        output_dir = self.eval_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        dataset = RippleUnlearningDataset(path=self.eval_cfg.data.ripple_unlearning.args.path)

        # 2. PREPARAÇÃO DOS ARTEFATOS COMPARTILHADOS
        # Para que os subprocessos funcionem, precisamos salvar o modelo base em disco (ou shared memory).
        # Salvar em disco é mais seguro para garantir o "clean state".
        temp_dir = os.path.join(output_dir, "temp_worker_artifacts")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        logger.info(f"💾 Saving base model state to {temp_dir} for worker distribution...")
        self.tokenizer.save_pretrained(temp_dir)
        model.config.save_pretrained(temp_dir)
        
        # Salvamos apenas os pesos. O worker recria a arquitetura via config.
        torch.save(model.state_dict(), os.path.join(temp_dir, "base_model.pt"))
        
        # 3. LIBERAÇÃO DE MEMÓRIA NO PROCESSO PAI
        # Isso é crucial: O processo pai não precisa mais do modelo na GPU.
        # Liberamos tudo para que o subprocesso tenha 100% da GPU disponível.
        logger.info("🗑️ Clearing Parent Process GPU memory...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        aggregated_results = defaultdict(list)
        detailed_results = []
        skipped_cases = 0

        # 4. LOOP DE MULTIPROCESSAMENTO
        # Usamos 'spawn' para garantir que o worker pegue um contexto CUDA virgem
        ctx = mp.get_context('spawn')
        
        for case_idx, case in enumerate(tqdm(dataset, desc="Evaluating (Isolated Processes)")):
            
            # Fila para receber os resultados do worker
            result_queue = ctx.Queue()
            
            # Criar e iniciar o processo do trabalhador
            p = ctx.Process(
                target=_run_case_in_worker,
                args=(temp_dir, case, self.eval_cfg, self.template_args, result_queue)
            )
            
            p.start()
            
            # Esperar resultado
            try:
                # Timeout generoso para evitar hangs eternos (ajuste conforme necessário)
                result = result_queue.get(timeout=None) 
            except Exception as e:
                logger.error(f"❌ Failed to get result from worker for case {case_idx}: {e}")
                result = None
            
            p.join() # Garante que o processo morreu e liberou recursos
            
            # Processar resultado
            if result and result.get("status") == "success":
                history = result["history"]
                epoch_0_metrics = result["epoch_0_metrics"]
                final_metrics = history[-1]["probes"]
                
                # --- AGGREGAÇÃO DE MÉTRICAS (Lógica original trazida para cá) ---
                clean_answers = {k: v for k,v in epoch_0_metrics["probes"].items()}
                case_res = {"case_id": case.get("case_id"), "probes": [], "history": history}

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
                
                # Salvar parciais periodicamente
                if case_idx % 5 == 0:
                    with open(os.path.join(output_dir, "ripple_unlearning_detailed_results.json"), 'w') as f:
                        json.dump(detailed_results, f, indent=4)
            else:
                skipped_cases += 1
                msg = result.get('message') if result else "Unknown error"
                logger.error(f"Skipping case {case_idx} due to worker failure: {msg}")

        # Cleanup final
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        final_results = {f"mean_{k}": sum(v)/len(v) if v else 0.0 for k,v in aggregated_results.items()}
        
        with open(os.path.join(output_dir, "ripple_unlearning_detailed_results.json"), 'w') as f:
            json.dump(detailed_results, f, indent=4)
        with open(os.path.join(output_dir, "ripple_unlearning_summary.json"), 'w') as f:
            json.dump(final_results, f, indent=4)
            
        return final_results