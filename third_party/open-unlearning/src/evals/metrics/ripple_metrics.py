# ruff: noqa
import logging
from typing import Any, Dict, List, Tuple

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
    max_length: int = 100,
) -> List[Tuple[bool, str]]:
    """
    Runs a list of probes against the model.
    Returns a list of tuples, where each tuple contains:
    (bool: whether a ground truth answer was found, str: the generated text)
    """
    if not probes:
        return []

    # Batch process all questions
    questions = [probe["question"] for probe in probes]
    inputs = tokenizer(
        questions, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_length, pad_token_id=tokenizer.eos_token_id
        )

    # Decode and check answers
    results = []
    for i, probe in enumerate(probes):
        # Slice the output to only get the generated part
        input_length = inputs["input_ids"][i].shape[0]
        generated_ids = outputs[i][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        has_answer = check_answers(generated_text, probe["answer"])
        results.append((has_answer, generated_text))

    return results

@unlearning_metric("forget_efficacy")
def forget_efficacy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probes: List[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Calculates the forgetting efficacy.
    This is the percentage of forget_probes where the model *fails* to generate the correct answer.
    A higher score is better.
    """
    if not probes:
        return {"forget_efficacy_rate": 1.0}

    logger = logging.getLogger(__name__)
    results = run_probes(model, tokenizer, probes)

    num_forgotten = 0
    for i, (has_answer, gen_text) in enumerate(results):
        if not has_answer:
            num_forgotten += 1

        logger.info(
            f"  Forget Probe: '{probes[i]['question']}' "
            f"-> Expected NOT in: {probes[i]['answer']}, Got: '{gen_text}' "
            f"-> {'✅ Forgotten' if not has_answer else '❌ Failed to Forget'}"
        )

    efficacy_rate = num_forgotten / len(results) if results else 1.0
    return {"forget_efficacy_rate": efficacy_rate}

@unlearning_metric("logical_inconsistency")
def logical_inconsistency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probes: List[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Calculates the logical inconsistency rate.
    This is the percentage of consistency_probes where the model *succeeds* in generating the correct answer,
    even though the premise was unlearned. A high score indicates high inconsistency.
    """
    if not probes:
        return {"logical_inconsistency_rate": 0.0}

    logger = logging.getLogger(__name__)
    results = run_probes(model, tokenizer, probes)

    num_inconsistent = 0
    for i, (has_answer, gen_text) in enumerate(results):
        if has_answer:
            num_inconsistent += 1

        logger.info(
            f"  Consistency Probe: '{probes[i]['question']}' "
            f"-> Expected: {probes[i]['answer']}, Got: '{gen_text}' "
            f"-> {'❌ Inconsistent' if has_answer else '✅ Consistent'}"
        )

    inconsistency_rate = num_inconsistent / len(results) if results else 0.0
    return {"logical_inconsistency_rate": inconsistency_rate}

@unlearning_metric("retain_accuracy")
def retain_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probes: List[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Calculates the retain accuracy.
    This is the percentage of retain_probes where the model *succeeds* in generating the correct answer.
    A higher score is better.
    """
    if not probes:
        return {"retain_accuracy_rate": 1.0}

    logger = logging.getLogger(__name__)
    results = run_probes(model, tokenizer, probes)

    num_retained = 0
    for i, (has_answer, gen_text) in enumerate(results):
        if has_answer:
            num_retained += 1

        logger.info(
            f"  Retain Probe: '{probes[i]['question']}' "
            f"-> Expected: {probes[i]['answer']}, Got: '{gen_text}' "
            f"-> {'✅ Retained' if has_answer else '❌ Forgotten'}"
        )

    accuracy_rate = num_retained / len(results) if results else 1.0
    return {"retain_accuracy_rate": accuracy_rate}    
