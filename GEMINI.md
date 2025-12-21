## Gemini Added Memories
- The user prefers to communicate in English.
- Code (variable names, methods, etc.) should be written in English.
- The user wants me to follow clean code principles and community best practices.
- Comments should be used sparingly, only for complex logic. Docstrings are preferred for documentation.

## Persona
I am a PhD-level LLM researcher with extensive experience at the world's leading AI labs (Google DeepMind, OpenAI, Anthropic, Meta, etc.). My current focus is on unlearning techniques in language models, and I am up-to-date with the state-of-the-art in this area. I will act as your research assistant, providing insights and support for your project.

## Research Context
The research objective is to investigate the "side effects" or "collateral damage" of state-of-the-art (SoTA) unlearning techniques. Specifically, the focus is to determine whether forcing a model to "forget" a specific fact also causes it to forget facts that are logically deduced from the original fact.

### Inspiration and Key Resources:
- **Inspirational Paper:** The research is inspired by the paper "Evaluating the Ripple Effects of Knowledge Editing in Language Models".
- **Benchmark to be Adapted:** The "RippleEdits" benchmark will be adapted to evaluate the effects of unlearning on logically connected facts.
- **Experimentation Framework:** The experiments will be built using the `open-unlearning` framework.

### Refined Core Hypothesis

Through discussion, the project's core hypothesis has been refined. It is not that unlearning causes widespread collateral damage (forgetting too much), but rather that it is **overly specific and fails to propagate logically**.

-   **The Problem:** Current unlearning methods will successfully forget a targeted fact (e.g., "X is the sibling of Y") but will *fail* to forget logically dependent facts (e.g., the inverse "Y is the sibling of X").
-   **The Consequence:** This creates a logically inconsistent knowledge base in the model, which is a significant and unevaluated issue.
-   **The Goal:** The `Ripple Unlearning Benchmark` is specifically designed to detect and quantify these logical inconsistencies.

### Relevant Artifacts:
- `@articles\Evaluating the Ripple Effects of Knowledge Editing in Language Models\**`
- `@RippleEdits\data\benchmark\**`
- `@open-unlearning-collateral-damage\docs\**`
- `@articles\OpenUnlearning Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics\**`

### Key Insights from Referenced Papers

#### "Evaluating the Ripple Effects of Knowledge Editing in Language Models"

This paper argues that conventional evaluation of knowledge editing (KE) is insufficient because it fails to account for logical implications. It introduces the concept of **"ripple effects,"** where editing one fact in an LLM should trigger consistent updates to other, logically related facts.

**Core Contributions:**

1.  **New Evaluation Criteria:** The paper proposes six criteria to measure these ripple effects:
    *   **Logical Generalization:** Whether logical constraints (e.g., symmetry, transitivity) are maintained post-edit.
    *   **Compositionality (I & II):** Whether the model can reason over the edited fact when composed with other facts.
    *   **Subject Aliasing:** Whether the edit propagates to aliases of the subject entity.
    *   **Preservation:** Whether other facts for a subject are preserved after editing a one-to-many relation.
    *   **Relation Specificity:** Whether unrelated facts about the subject entity remain unaffected.

2.  **RippleEdits Benchmark:** A diagnostic benchmark with approximately 5,000 examples designed to test these six criteria.

**Key Findings:**

*   Existing KE methods (e.g., ROME, MEMIT, MEND) often fail to produce consistent ripple effects, suggesting their edits are superficial.
*   A simple In-Context Editing (ICE) baseline, which provides the edit as context in the prompt, outperforms parametric methods.
*   Performance on ripple effect tasks generally improves with model scale.

#### "OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics"

This paper introduces **`OpenUnlearning`**, a standardized framework to address the fragmentation in LLM unlearning research by unifying the benchmarking of methods and metrics.

**Core Contributions:**

1.  **`OpenUnlearning` Framework:** A unified, open-source library that integrates:
    *   13+ unlearning algorithms.
    *   16+ evaluation metrics.
    *   3 major benchmarks: TOFU, MUSE, and WMDP.
    *   It is designed to be modular, simplifying the integration of new components and the reproduction of experiments.

2.  **Meta-Evaluation of Metrics:** The paper proposes a framework to evaluate unlearning metrics themselves against two key properties:
    *   **Faithfulness:** A metric's ability to accurately detect the presence or absence of specific knowledge.
    *   **Robustness:** A metric's stability when the model is subjected to stress tests like quantization or relearning.
    *   *Finding:* Extraction Strength (ES) and Exact Memorization (EM) were identified as the most reliable metrics.

3.  **Method Benchmarking:** The framework is used to conduct a large-scale comparison of 8 unlearning methods, finding that **SimNPO** and **RMU** are strong performers, although rankings are sensitive to how different evaluation aspects (e.g., forgetting, privacy, utility) are weighted.

### OpenUnlearning Framework: A Practical Guide

The `OpenUnlearning` framework is a standardized, modular library for benchmarking LLM unlearning methods. It is built on `PyTorch` and heavily utilizes `Hydra` for configuration.

**Core Principles:**

*   **Standardization:** Provides a unified structure for implementing and comparing unlearning methods, datasets, and evaluation metrics.
*   **Extensibility:** New components (trainers, datasets, metrics) can be added by following a "handler, registry, config" pattern:
    1.  **Implement a Handler:** A Python class/function with the core logic.
    2.  **Register the Handler:** Add it to a registry to make it accessible by a key.
    3.  **Create a Config:** A YAML file that points to the handler and sets its parameters.
*   **Configuration with Hydra:** Experiments are defined by composing YAML configuration files. A main config (`unlearn.yaml` or `eval.yaml`) is combined with an `experiment` config (e.g., `experiment/unlearn/tofu/default.yaml`) and can be modified with command-line overrides.

**Available Components:**

*   **Benchmarks:** TOFU, MUSE, WMDP.
*   **Unlearning Methods:** GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL, and more.
*   **Models:** Llama family, Phi, Gemma, Zephyr, and others.

#### Running Experiments

**1. Setup Environment**

```bash
# Create conda environment
conda create -n unlearning python=3.11
conda activate unlearning

# Install dependencies
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3

# Download reference evaluation data
python setup_data.py --eval
```

**2. Perform Unlearning**

Unlearning jobs are run using `src/train.py`. The configuration is controlled via the command line.

*Example: Run `GradAscent` on the TOFU `forget10` split.*
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=GradAscent \
  forget_split=forget10 \
  retain_split=retain90 \
  task_name=my_unlearning_run
```
*   `experiment`: Specifies the base configuration for the benchmark (datasets, model, etc.).
*   `trainer`: Overrides the default unlearning method.
*   `task_name`: Defines the unique output directory for this run.

**3. Perform Evaluation**

Evaluations are run using `src/eval.py`.

*Example: Evaluate the model saved from the previous run.*
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/my_unlearning_run \
  task_name=my_unlearning_run_eval
```
*   `model.model_args.pretrained_model_name_or_path`: Points to the checkpoint of the model to be evaluated.
*   `retain_logs_path`: (If needed by a metric) points to a JSON file with evaluation results from a reference "retain" model.

---

### Plan: A Custom Evaluator for Ripple Unlearning

This plan outlines the most robust and framework-aligned method for achieving our research goals.

#### **Phase 1: Data Preparation**

1.  **Create `prepare_ripple_unlearning_benchmark.py` Script:**
    *   **Location**: `src/data_preparation/`
    *   **Logic**: Convert `RippleEdits` JSON data into a new `ripple_unlearning_benchmark.json`. Each new record will contain:
        *   `case_id`, `forget_request`, `forget_probes`, `consistency_probes`, and `retain_probes`.

#### **Phase 2: Framework Integration (within `third_party/open-unlearning`)**

1.  **Create a `Dataset` Handler (`src/data/ripple_dataset.py`):**
    *   Implement `RippleUnlearningDataset` to load our new benchmark file. `__getitem__` will return a full record.
    *   Register the handler.

2.  **Create Custom Metric Handlers (`src/evals/metrics/ripple_metrics.py`):**
    *   Implement stateless functions like `logical_consistency(model, ...)` and `forget_efficacy(model, ...)` to compute scores for a given set of probes.
    *   Register the handlers.

3.  **Create the Custom Benchmark/Evaluator (`src/evals/ripple_unlearning_evaluator.py`):**
    *   This is the core of the plan, orchestrating the one-by-one evaluation loop.
    *   **Logic**: The `evaluate` method will:
        1.  Save a clean copy of the initial model state.
        2.  Loop through every fact in the `RippleUnlearningDataset`.
        3.  In each loop:
            a. **Restore** the model to its clean state.
            b. **Unlearn** the single fact by calling the configured `trainer.train()` on a temporary, single-item dataset.
            c. **Evaluate** the modified model using the custom metric handlers on the corresponding probes for that fact.
            d. **Aggregate** the results.
        4.  Return the final, averaged results.
    *   Register the handler.

#### **Phase 3: Configuration & Execution**

1.  **Create Config Files**: Define YAML files for the new dataset, metrics, and evaluator in the `configs/` directory. Create a main experiment config at `configs/experiment/eval/ripple_unlearning/default.yaml` that brings them all together.

2.  **Run Experiment**: Use `src/eval.py` with our new experiment config. Different unlearning algorithms can be tested by simply changing the `trainer` argument at the command line.

---
### Ripple Unlearning Evaluation Strategy

This project adapts the `RippleEdits` evaluation methodology for an unlearning context. The core idea is to test one fact at a time to isolate its specific ripple effects.

**Evaluation Flow (per-fact):**

1.  **Isolate Fact**: For each entry in our benchmark, the process is performed independently.
2.  **Unlearn**: A single fact is forgotten using an unlearning algorithm.
3.  **Test Probes**: The modified model is tested against several sets of probes:
    *   **Forget Efficacy Probes**: Verify the original fact was forgotten.
    *   **Logical Consistency Probes**: Test if logically entailed facts were also forgotten. A failure here (i.e., the model still knows the entailed fact) is a finding that supports our hypothesis.
    *   **Retain Probes**: Test if unrelated facts are preserved to measure catastrophic forgetting.
4.  **Restore Model**: The model is reset to its original, clean state before processing the next fact.

**Proposed Primary Metrics:**

1.  **Forgetting Efficacy Rate**: The percentage of times the model successfully forgets the target fact. A high score is a prerequisite for other metrics.
2.  **Logical Inconsistency Rate**: The percentage of times the model *correctly* answers a query for a logically entailed fact, *after* the premise was supposed to have been unlearned. A high score here supports our core hypothesis. This is the main metric for our study.
3.  **Retain Accuracy**: The percentage of unrelated facts that are correctly recalled. This is a standard utility metric to guard against catastrophic forgetting.

---
### Experimental Plan for Publication

This plan is designed to produce a comprehensive study suitable for a research paper, investigating the logical inconsistency of unlearning.

**Experiment 1: Baseline Logical Inconsistency of SOTA Methods**
*   **Research Question:** To what extent do current state-of-the-art (SoTA) unlearning methods fail to propagate forgetting to logically entailed facts?
*   **Methodology:**
    1.  Select a representative set of unlearning algorithms from `open-unlearning` (e.g., a baseline like Gradient Ascent, and SOTA methods like NPO, SimNPO, RMU).
    2.  Run the `RippleUnlearningEvaluator` for each algorithm on the full `ripple_unlearning_benchmark`.
*   **Data & Metrics:**
    *   **Primary Metric:** Logical Inconsistency Rate.
    *   **Secondary Metrics:** Forgetting Efficacy Rate, Retain Accuracy, and the holistic `Model Utility` score from the TOFU benchmark.
*   **Expected Contribution:** This experiment will provide the main result for the paper: a quantitative comparison demonstrating that current methods consistently fail to maintain logical consistency after unlearning.

**Experiment 2: Probabilistic and Privacy-Based Analysis of Inconsistency**
*   **Research Question:** Is the logical inconsistency a surface-level artifact, or does it persist at a probabilistic and information-theoretic level?
*   **Methodology:**
    1.  Use the unlearned models from Experiment 1.
    2.  Apply more nuanced metrics available in `open-unlearning`.
*   **Data & Metrics:**
    *   **Adapted Truth Ratio**: Apply the `Truth Ratio` metric (from TOFU) to our `consistency_probes`. A high ratio (model is confident in the entailed fact) provides stronger evidence of inconsistency than simple accuracy.
    *   **Membership Inference Attacks (MIA)**: Apply MIA metrics (e.g., LOSS attack from MUSE) to the `consistency_probes`. If an MIA can determine that a logically entailed fact `F'` was "known" by the original model, it implies that traces of the forgotten fact `F` remain, revealing a new privacy vulnerability.
*   **Expected Contribution:** This deepens the analysis beyond simple accuracy, showing that the logical inconsistency is a fundamental failure of the unlearning process, with implications for model-internal knowledge representations and privacy.

**Experiment 3: Analysis of Causal Factors (Relation Type & Entity Popularity)**
*   **Research Question:** Which factors influence the rate of logical inconsistency? Is it harder for models to consistently forget facts about popular entities or certain types of logical relationships?
*   **Methodology:**
    1.  The `RippleUnlearningBenchmark` will have metadata for each entry, including the `RippleEdits` criteria (`Logical_Generalization`, `Compositionality_I`, etc.) and the entity popularity bucket (`Recent`, `Random`, `Popular`).
    2.  Analyze the results from Experiment 1 by grouping them based on this metadata.
*   **Data & Metrics:**
    *   Break down the **Logical Inconsistency Rate** by:
        *   **Ripple Effect Type:** Compare rates for symmetric relations (`Logical_Generalization`), two-hop reasoning (`Compositionality_I/II`), and `Subject_Aliasing`.
        *   **Entity Popularity:** Compare rates for facts from the `Recent`, `Random`, and `Popular` subsets of `RippleEdits`.
*   **Expected Contribution:** This provides valuable insights into *why* and *when* unlearning fails to be logically consistent. For example, it might show that unlearning complex, multi-hop compositional facts is significantly harder than unlearning simple symmetric ones, or that unlearning facts about highly-interconnected, popular entities is more prone to leaving behind logical contradictions. This analysis is directly inspired by the findings in the original `RippleEdits` paper (e.g., Table 2 and Figure 7).