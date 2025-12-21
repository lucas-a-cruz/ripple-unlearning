# ruff: noqa
from typing import Any, Dict, List

import torch
from evals.metrics.base import unlearning_metric
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_answers(generated_text: str, answer_list: List[str]) -> bool:
    """
    Checks if any of the answers in the answer_list are present in the generated_text.
    The check is case-insensitive.
    """
    generated_text_lower = generated_text.lower()
    for answer in answer_list:
        if answer.lower() in generated_text_lower:
            return True
    return False

def run_probes(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    probes: List[Dict[str, Any]],
    max_length: int = 100
) -> List[bool]:
    """
    Runs a list of probes against the model and returns a list of boolean success values.
    """
    if not probes:
        return []

    # Batch process all questions
    questions = [probe["question"] for probe in probes]
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and check answers
    results = []
    for i, probe in enumerate(probes):
        # Slice the output to only get the generated part
        input_length = inputs['input_ids'][i].shape[0]
        generated_ids = outputs[i][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        has_answer = check_answers(generated_text, probe["answer"])
        results.append(has_answer)
        
    return results

@unlearning_metric("forget_efficacy")
def forget_efficacy(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    probes: List[Dict[str, Any]], 
    **kwargs: Any
) -> Dict[str, float]:
    """
    Calculates the forgetting efficacy.
    This is the percentage of forget_probes where the model *fails* to generate the correct answer.
    A higher score is better.
    """
    if not probes:
        return {"forget_efficacy_rate": 1.0}

    results = run_probes(model, tokenizer, probes)
    # Efficacy is high if the answer is NOT found
    num_forgotten = results.count(False)
    efficacy_rate = num_forgotten / len(results)
    
    return {"forget_efficacy_rate": efficacy_rate}

@unlearning_metric("logical_inconsistency")
def logical_inconsistency(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    probes: List[Dict[str, Any]], 
    **kwargs: Any
) -> Dict[str, float]:
    """
    Calculates the logical inconsistency rate.
    This is the percentage of consistency_probes where the model *succeeds* in generating the correct answer,
    even though the premise was unlearned. A high score indicates high inconsistency.
    """
    if not probes:
        return {"logical_inconsistency_rate": 0.0}

    results = run_probes(model, tokenizer, probes)
    # Inconsistency is high if the answer IS found
    num_inconsistent = results.count(True)
    inconsistency_rate = num_inconsistent / len(results)
    
    return {"logical_inconsistency_rate": inconsistency_rate}

@unlearning_metric("retain_accuracy")
def retain_accuracy(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    probes: List[Dict[str, Any]], 
    **kwargs: Any
) -> Dict[str, float]:
    """
    Calculates the retain accuracy.
    This is the percentage of retain_probes where the model *succeeds* in generating the correct answer.
    A higher score is better.
    """
    if not probes:
        return {"retain_accuracy_rate": 1.0}

    results = run_probes(model, tokenizer, probes)
    # Accuracy is high if the answer IS found
    num_retained = results.count(True)
    accuracy_rate = num_retained / len(results)
    
    return {"retain_accuracy_rate": accuracy_rate}    
