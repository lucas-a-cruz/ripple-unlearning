# Domain-Specific Unlearning Pipeline

This directory contains a complete pipeline for generating domain-specific content and performing unlearning experiments. The pipeline integrates the `domain-generation` module with the OpenUnlearning framework to enable unlearning of arbitrary knowledge domains.

## 🚀 Quick Start

```bash
# Run the complete pipeline for a topic
bash scripts/domain-unlearn.sh "Brazil"

# Use a different model and unlearning method
bash scripts/domain-unlearn.sh "USA History" Llama-3.2-3B-Instruct GradAscent

# Try different topics
bash scripts/domain-unlearn.sh "Mexican Food" Llama-3.1-8B-Instruct NPO
```

## 📋 Overview

The domain unlearning pipeline consists of the following steps:

1. **Domain Content Generation**: Uses LangGraph-based LLM agents to generate comprehensive content about a topic (books, articles, QA pairs)
2. **Dataset Conversion**: Converts generated content into HuggingFace dataset format compatible with OpenUnlearning
3. **Configuration Setup**: Creates Hydra configuration files for the unlearning experiment
4. **Unlearning**: Runs the unlearning algorithm on a pre-trained model
5. **Evaluation**: (Optional) Evaluates the unlearned model

## 🗂️ Project Structure

```
.
├── src/
│   └── domain_generation/          # Domain content generation module
│       ├── main.py                 # Entry point for generation
│       ├── convert_to_dataset.py   # Converts domain.json to HF datasets
│       ├── config.py               # Generation configuration
│       ├── models.py               # Pydantic models for data structures
│       ├── graphs/                 # LangGraph workflow definitions
│       ├── prompts/                # LLM prompt templates
│       └── utils.py                # Utility functions
│
├── scripts/
│   └── domain-unlearn.sh           # Main pipeline script
│
├── data/
│   └── run/                        # Run-specific data (generated)
│       └── {timestamp}/
│           ├── {topic}/
│           │   ├── qa_dataset/     # QA pairs for forget/retain
│           │   ├── text_dataset/   # Full text for pretraining-style unlearning
│           │   └── metadata.json   # Dataset statistics
│           └── run_summary.json    # Run configuration and paths
│
├── output/                         # Domain generation outputs
│   └── {timestamp}/
│       └── domain.json             # Generated domain content
│
├── saves/
│   └── unlearn/
│       └── {run_name}/             # Model checkpoints and evaluation results
│
└── configs/
    ├── data/datasets/              # Dataset configs (auto-generated)
    │   ├── DOMAIN_{topic}_forget.yaml
    │   └── DOMAIN_{topic}_retain.yaml
    └── experiment/unlearn/domain/  # Experiment configs (auto-generated)
        └── {topic}.yaml
```

## 🔧 Setup

### Prerequisites

1. **Environment Setup**:
   ```bash
   conda create -n unlearning python=3.11
   conda activate unlearning
   pip install .[lm_eval]
   pip install --no-build-isolation flash-attn==2.6.3
   ```

2. **API Keys**:
   Create a `.env` file in the project root with your API keys:
   ```bash
   # OpenAI API (for domain generation)
   OPENAI_API_KEY=your_openai_api_key_here

   # Optional: Anthropic API
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

3. **Data Setup** (for evaluation):
   ```bash
   python setup_data.py --eval
   ```

### Domain Generation Configuration

You can customize the domain generation in `src/domain_generation/config.py`:

```python
class GenerationConfig(BaseSettings):
    # Domain and Topic Configuration
    topics_min_items: int = 2         # Minimum number of topics
    topics_max_items: int = 5         # Maximum number of topics
    articles_min_per_topic: int = 2   # Articles per topic

    # Book Configuration
    toc_min_items: int = 2            # Min chapters per book
    toc_max_items: int = 4            # Max chapters per book
    sections_min_per_chapter: int = 2 # Min sections per chapter
    sections_max_per_chapter: int = 4 # Max sections per chapter

    # Article Configuration
    sections_min_per_article: int = 3
    sections_max_per_article: int = 5

    # QA Configuration
    grounded_qa_min_items: int = 5    # QA pairs answerable from content
    grounded_qa_max_items: int = 10
    ungrounded_qa_min_items: int = 3  # QA pairs NOT answerable from content
    ungrounded_qa_max_items: int = 5

    # LLM Configuration
    model_name: str = "gpt-5-mini"
    temperature: float = 0.7
    max_retries: int = 5
```

You can override these via environment variables with the `GEN_` prefix:
```bash
export GEN_TOPICS_MIN_ITEMS=3
export GEN_TOPICS_MAX_ITEMS=6
export GEN_GROUNDED_QA_MIN_ITEMS=10
```

## 🎯 Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
bash scripts/domain-unlearn.sh "Brazil"
```

This will:
1. Generate content about Brazil (topics, books, articles, QA pairs)
2. Convert to HuggingFace datasets
3. Create configuration files
4. Run unlearning with `Llama-3.2-1B-Instruct` and `GradAscent`
5. Save the unlearned model checkpoint

### Advanced Usage

Specify custom model and unlearning method:

```bash
bash scripts/domain-unlearn.sh "USA History" Llama-3.2-3B-Instruct NPO
```

Available models:
- `Llama-3.2-1B-Instruct`
- `Llama-3.2-3B-Instruct`
- `Llama-3.1-8B-Instruct`
- (See `configs/model/` for full list)

Available unlearning methods:
- `GradAscent` - Simple gradient ascent (loss negation)
- `GradDiff` - Gradient difference
- `NPO` - Negative Preference Optimization
- `SimNPO` - Simplified NPO
- `DPO` - Direct Preference Optimization
- `RMU` - Representation-level unlearning
- `PDU` - Parametric Data Unlearning
- (See `configs/trainer/` for full list)

### Manual Step-by-Step Execution

If you prefer to run steps individually:

#### Step 1: Generate Domain Content

```bash
# Using the default main.py
python -m src.domain_generation.main

# Or run with custom topic via Python
python -c "
from src.domain_generation.graphs import build_domain_graph
import json
from pathlib import Path

domain_graph = build_domain_graph()
result = domain_graph.invoke({
    'name': 'Your Topic',
    'description': 'Description of your topic'
})

# Save output
output_dir = Path('output/my_run')
output_dir.mkdir(exist_ok=True, parents=True)
with open(output_dir / 'domain.json', 'w') as f:
    json.dump(result['domain'].model_dump(), f, indent=2)
"
```

#### Step 2: Convert to Dataset

```bash
python -m src.domain_generation.convert_to_dataset \
    output/my_run/domain.json \
    --output-dir data/domain_datasets \
    --dataset-name my_topic \
    --split-ratio 0.8
```

This creates:
- `data/domain_datasets/my_topic/qa_dataset/` - QA pairs split into forget/retain
- `data/domain_datasets/my_topic/text_dataset/` - Full text split into forget/retain
- `data/domain_datasets/my_topic/metadata.json` - Dataset statistics

#### Step 3: Create Config Files

Create dataset configs in `configs/data/datasets/`:

**DOMAIN_my_topic_forget.yaml**:
```yaml
DOMAIN_my_topic_forget:
  handler: QADataset
  args:
    hf_args:
      path: "data/domain_datasets/my_topic/qa_dataset"
      split: "forget"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
```

**DOMAIN_my_topic_retain.yaml**:
```yaml
DOMAIN_my_topic_retain:
  handler: QADataset
  args:
    hf_args:
      path: "data/domain_datasets/my_topic/qa_dataset"
      split: "retain"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
```

#### Step 4: Run Unlearning

```bash
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/domain/my_topic \
    trainer=GradAscent \
    model=Llama-3.2-1B-Instruct \
    task_name=my_topic_unlearn \
    trainer.args.num_train_epochs=5 \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=4
```

#### Step 5: Evaluate (Optional)

```bash
python src/eval.py --config-name=eval.yaml \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/my_topic_unlearn \
    task_name=my_topic_eval
```

## 📊 Understanding the Output

### Domain Content Generation

The `domain.json` file contains the complete generated domain structure:

```json
{
  "name": "Brazil",
  "description": "Brazilian culture, history, geography, and society",
  "topics": [
    {
      "name": "Physical Geography and Environment",
      "description": "...",
      "idx": 1
    }
  ],
  "books": [
    {
      "title": "Brazil: Landscapes, Climate and the Environment",
      "topic": "Physical Geography and Environment",
      "chapters": [...],
      "grounded_questions": [
        {
          "question": "Which soil order is dominant...",
          "answer": "Oxisols",
          "is_grounded": true
        }
      ],
      "ungrounded_questions": [...]
    }
  ],
  "articles": [...]
}
```

### Dataset Formats

**QA Dataset** (`qa_dataset/`):
- Format: `{"question": str, "answer": str, "source": str, "topic": str}`
- Used for: Question-answering style unlearning (similar to TOFU benchmark)
- Split: 80% forget, 20% retain (configurable)

**Text Dataset** (`text_dataset/`):
- Format: `{"text": str, "source": str, "topic": str}`
- Used for: Pretraining-style unlearning (similar to MUSE benchmark)
- Split: 80% forget, 20% retain (configurable)

### Model Checkpoints

After unlearning, you'll find:
```
saves/unlearn/{run_name}/
├── checkpoint-{step}/          # Periodic checkpoints
│   ├── pytorch_model.bin
│   ├── config.json
│   └── training_args.bin
├── evals/                      # Evaluation results (if run)
├── trainer_state.json          # Training state
└── ...
```

## 🔬 Experimental Variations

### Different Data Sizes

Adjust the split ratio to experiment with different forget/retain proportions:

```bash
python -m src.domain_generation.convert_to_dataset \
    output/my_run/domain.json \
    --split-ratio 0.9  # 90% forget, 10% retain
```

### Pretraining-Style Unlearning

To use full text instead of QA pairs, modify the dataset handler in your config:

```yaml
DOMAIN_my_topic_forget:
  handler: PretrainingDataset
  args:
    hf_args:
      path: "data/domain_datasets/my_topic/text_dataset"
      split: "forget"
    text_key: "text"
    max_length: 2048
```

### Hyperparameter Tuning

Modify training hyperparameters in the script or via command line:

```bash
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/domain/my_topic \
    trainer.args.learning_rate=5e-6 \
    trainer.args.num_train_epochs=10 \
    trainer.args.warmup_epochs=2.0
```

## 📈 Evaluation and Analysis

### Testing Unlearning Success

After unlearning, test the model to verify it has "forgotten" the domain:

1. **Direct QA Testing**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   # Load unlearned model
   model = AutoModelForCausalLM.from_pretrained("saves/unlearn/my_run")
   tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.2-1B-Instruct")

   # Test with questions from forget set
   question = "What is the capital of Brazil?"
   inputs = tokenizer(question, return_tensors="pt")
   outputs = model.generate(**inputs)
   print(tokenizer.decode(outputs[0]))
   ```

2. **Compare with Baseline**:
   - Test the same questions on the original model
   - Measure changes in confidence, accuracy, or answer quality

3. **Retention Testing**:
   - Test on retain set to ensure model still performs well on retained knowledge
   - Verify no catastrophic forgetting of unrelated capabilities

### Evaluation Metrics

For comprehensive evaluation, integrate with OpenUnlearning's evaluation framework:

```bash
python src/eval.py \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/my_run \
    # Add custom evaluation metrics here
```

Key metrics to monitor:
- **Forget Quality**: How well the model has forgotten the target domain
- **Model Utility**: Performance on retained knowledge
- **General Capability**: Performance on standard benchmarks (MMLU, etc.)

## 🐛 Troubleshooting

### Common Issues

1. **API Rate Limits**:
   - The domain generation uses LLM APIs which may have rate limits
   - The system includes retry logic with exponential backoff
   - If generation fails, check your API key and quota

2. **Out of Memory**:
   - Reduce batch size: `trainer.args.per_device_train_batch_size=2`
   - Enable gradient checkpointing: `trainer.args.gradient_checkpointing=true`
   - Use gradient accumulation: `trainer.args.gradient_accumulation_steps=8`

3. **Generation Quality**:
   - Adjust temperature: Set `GEN_TEMPERATURE=0.5` for more focused content
   - Modify quantity ranges in `src/domain_generation/config.py`
   - Review generated content in `output/{timestamp}/domain.json`

4. **Dataset Issues**:
   - Ensure the split ratio is between 0 and 1
   - Check that the domain.json file is valid JSON
   - Verify the output directory has write permissions

### Debug Mode

Enable verbose logging:

```bash
export LOGURU_LEVEL=DEBUG
bash scripts/domain-unlearn.sh "My Topic"
```

## 📚 Examples

### Example 1: Unlearning Brazilian Geography

```bash
bash scripts/domain-unlearn.sh "Brazil" Llama-3.2-1B-Instruct GradAscent
```

Expected output:
- ~2-4 topics (Geography, History, Culture, etc.)
- ~2-4 books with multiple chapters
- ~4-8 articles
- ~40-80 QA pairs total

### Example 2: Unlearning US History

```bash
bash scripts/domain-unlearn.sh "USA History" Llama-3.2-3B-Instruct NPO
```

This will generate content covering various aspects of US history and unlearn using NPO method.

### Example 3: Unlearning Cuisine Knowledge

```bash
bash scripts/domain-unlearn.sh "Mexican Food" Llama-3.1-8B-Instruct DPO
```

Generates culinary knowledge about Mexican food and performs unlearning.

## 🤝 Contributing

To extend or modify the pipeline:

1. **Add New Unlearning Methods**: See `src/trainer/unlearn/`
2. **Modify Generation**: Edit `src/domain_generation/graphs/` and `src/domain_generation/prompts/`
3. **Custom Evaluation**: Add metrics in `src/evals/metrics/`

## 📖 References

- [OpenUnlearning Paper](https://arxiv.org/abs/2506.12618)
- [TOFU Benchmark](https://arxiv.org/abs/2401.06121)
- [MUSE Benchmark](https://arxiv.org/abs/2407.06460)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## 📝 License

This project inherits the MIT License from the OpenUnlearning framework.

---

**Questions or Issues?** Please open an issue in the repository or refer to the main OpenUnlearning documentation.
