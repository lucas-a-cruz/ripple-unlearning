# Ripple Unlearning: Investigating the Collateral Damage of Forgetting

This project investigates the "collateral damage" of machine unlearning. Our goal is to determine if forcing a Large Language Model (LLM) to forget a specific fact also causes it to forget other, logically related facts—a phenomenon we term "ripple effects" in the context of unlearning.

To achieve this, we adapt the `RippleEdits` benchmark, originally designed for knowledge editing, into a new benchmark for evaluating unlearning. The experiments are conducted using the `open-unlearning` framework.

## Core Hypothesis

Our central hypothesis is that current unlearning methods are overly specific and fail to produce consistent logical "ripple effects." We believe that when a model is instructed to forget a fact, it will successfully erase that specific piece of information but fail to unlearn other facts that are logically dependent on it, such as those involving symmetric relations. This leads to a logically inconsistent knowledge base.

For instance, if a model is taught to forget that "X is the sibling of Y", we predict it will not automatically forget the inverse and will continue to affirm that "Y is the sibling of X".

This demonstrates that the unlearning is superficial and fails to remove the underlying concept, creating contradictions. This is a significant and largely unevaluated problem in current unlearning research, which our benchmark is designed to detect and quantify.

## Methodology: Adapting RippleEdits for Unlearning

The `RippleEdits` benchmark was created to evaluate whether knowledge *editing* produces consistent logical "ripple effects." We adapt its principles to evaluate knowledge *unlearning*. Instead of checking if an *edit* propagates logically, we check if *forgetting* a fact also erases logically connected knowledge.

The original six evaluation criteria from `RippleEdits` are mapped to our unlearning scenario as follows:

1.  **Logical Generalization**: If we unlearn a fact with a symmetric or transitive relation (e.g., `(A, sibling of, B)`), does the model also forget the inverse relation (e.g., `(B, sibling of, A)`)?
2.  **Compositionality**: If we make the model forget a fact that is part of a multi-hop reasoning chain (e.g., unlearn `(A, is capital of, B) ` where `B` is a country in `C`), does the model also lose the ability to answer compositional questions (e.g., "What is the capital of the country B in continent C?")?
3.  **Subject Aliasing**: When we unlearn a fact about a subject (e.g., "Joe Biden"), does the model also forget the same fact when the subject is referred to by an alias (e.g., "the 46th U.S. President")?
4.  **Preservation**: If we unlearn one fact from a one-to-many relationship (e.g., one child of a person with multiple children), does the model retain knowledge of the other, untargeted facts (e.g., the other children)?
5.  **Relation Specificity**: After unlearning a specific fact about an entity (e.g., `(Entity, relation1, value1)`), does the model preserve knowledge of unrelated facts (e.g., `(Entity, relation2, value2)`)? This measures the precision of the unlearning method.

## The Ripple Unlearning Benchmark

The script `src/data_preparation/prepare_ripple_unlearning_benchmark.py` transforms the original `RippleEdits` dataset into a format compatible with the `open-unlearning` framework. The original benchmark, designed for knowledge editing, contains entries with a requested edit and a set of evaluation queries. We repurpose these components as follows:

-   **Forget Set**: The fact to be unlearned is the *original, pre-edit fact* from each `RippleEdits` entry. For example, if `RippleEdits` proposes editing `(A, capital of, B)` to `(A, capital of, C)`, our forget set will include the fact `(A, capital of, B)`.

-   **Evaluation Probes**: The evaluation queries associated with each entry in `RippleEdits` are repurposed as our probes. These queries are already designed to test for logical consistency post-edit, making them ideal for testing logical consistency post-unlearning. Each probe checks if the model's knowledge remains logically consistent after a related fact is forgotten, targeting the five criteria outlined in our methodology.

-   **Retain Set**: To monitor for catastrophic forgetting, a retain set is constructed from facts within the `RippleEdits` benchmark that are not selected for the forget set in a given experiment and share no entities with them. This ensures that the model's general utility remains intact while unlearning specific, targeted facts.

The final output is saved in `data/processed/ripple_unlearning_benchmark` and contains these three distinct sets of data for each experiment.

## Custom Evaluation Workflow

The entire process is designed to test our core hypothesis by unlearning one fact at a time and measuring the "ripple effects." This is orchestrated by a custom evaluator class within the `open-unlearning` framework.

#### 1. The Trigger Command

It all starts with a single command line instruction from within the `third_party/open-unlearning/` directory:
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/ripple_unlearning/default \
  trainer=GradAscent \
  data.ripple_unlearning.args.path=data/processed/ripple_unlearning_benchmark/recent.jsonl \
  task_name=ripple_eval_recent_grad_ascent
```
The `experiment=eval/ripple_unlearning/default` argument loads our custom configuration, which points to the `RippleUnlearningDataset` handler and the `RippleUnlearningEvaluator`.

#### 2. The `RippleUnlearningEvaluator` Takes Control

The `open-unlearning` framework instantiates our `RippleUnlearningEvaluator` and calls its `evaluate()` method. This method contains the core logic for our experiment and performs the following loop:

1.  **Save Clean State**: Before the loop begins, the evaluator saves a "clean" copy of the original model's weights to a temporary file.
2.  **Iterate Over Benchmark**: It then loops through every single fact (every line) in our benchmark dataset, one by one.

**Inside the Loop (For each fact):**

3.  **Restore Model**: The clean model state is loaded from the file, ensuring the unlearning of one fact does not interfere with the evaluation of the next.

4.  **Perform Unlearning**: The evaluator calls the `train()` method of the specified unlearning algorithm (e.g., `GradAscent`). To do this, it dynamically creates a temporary `ForgetRetainDataset` containing:
    -   **A "forget" set**: The single fact to be unlearned for the current case.
    -   **A "retain" set**: The list of unrelated facts (`retain_probes`) for that case, which tells the algorithm what to preserve.
    The trainer then runs its optimization process on this tiny, single-case dataset.

5.  **Measure Ripple Effects**: Immediately after the model is modified, the evaluator calls our custom metric functions (`forget_efficacy`, `logical_inconsistency`, `retain_accuracy`) on the corresponding probes (`forget_probes`, `consistency_probes`, `retain_probes`). These functions check if the expected answer is present in the model's generated text.

6.  **Store Results**: The results from this single fact (e.g., efficacy: 1.0, inconsistency: 0.8) are stored, and the loop proceeds to the next fact, starting again by restoring the clean model.

#### 3. Final Report

After the loop completes, the evaluator calculates the `mean` of all collected scores and returns a final dictionary, which is printed to the console:
```json
{
  "mean_forget_efficacy_rate": 0.98,
  "mean_logical_inconsistency_rate": 0.85,
  "mean_retain_accuracy_rate": 0.96
}
```

## Folder Structure

```
/
├── README.md                  # This file
├── articles/                  # Research papers and related materials.
├── third_party/               # External codebases that are not modified.
│   ├── RippleEdits/           # The original RippleEdits benchmark code.
│   └── open-unlearning/       # The open-unlearning framework code.
│
├── data/                      # All data, raw and processed.
│   ├── raw/                   # Raw, unmodified data.
│   └── processed/             # Processed data ready for experiments.
│
├── experiments/               # Experiment-specific files.
│   ├── configs/               # Hydra configs for our experiments.
│   └── scripts/               # Scripts to run experiments.
│
├── src/                       # Custom source code for this project.
│   └── data_preparation/      # Scripts for data processing.
│
├── results/                   # Outputs from experiments (models, logs, evaluations).
│
└── archive/                   # Archived miscellaneous files.
```

