#!/bin/bash

##############################################################################
# Domain Unlearning Pipeline
#
# This script performs end-to-end domain unlearning:
# 1. Generates domain content (books, articles, QA) for a specified topic
# 2. Converts the generated content to HuggingFace dataset format
# 3. Runs unlearning on a specified model
# 4. Evaluates the unlearned model
#
# Usage:
#   bash scripts/domain-unlearn.sh <TOPIC> [MODEL] [TRAINER]
#
# Example:
#   bash scripts/domain-unlearn.sh "Brazil"
#   bash scripts/domain-unlearn.sh "USA History" Llama-3.2-3B-Instruct GradAscent
#   bash scripts/domain-unlearn.sh "Mexican Food" Llama-3.1-8B-Instruct NPO
##############################################################################

set -e  # Exit on error

# Parse command-line arguments
TOPIC="${1:-Brazil}"
MODEL="${2:-Llama-3.2-1B-Instruct}"
TRAINER="${3:-GradAscent}"

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/${TIMESTAMP}"
DATA_DIR="data/run/${TIMESTAMP}"
DATASET_NAME=$(echo "${TOPIC}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
RUN_NAME="${DATASET_NAME}_${TIMESTAMP}"

# Training hyperparameters
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=5
LEARNING_RATE=1e-5

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "================================================================================================"
echo "Domain Unlearning Pipeline"
echo "================================================================================================"
echo "Topic:                ${TOPIC}"
echo "Model:                ${MODEL}"
echo "Trainer:              ${TRAINER}"
echo "Dataset Name:         ${DATASET_NAME}"
echo "Run Name:             ${RUN_NAME}"
echo "Output Directory:     ${OUTPUT_DIR}"
echo "Data Directory:       ${DATA_DIR}"
echo "Timestamp:            ${TIMESTAMP}"
echo "================================================================================================"
echo ""

##############################################################################
# Step 1: Generate Domain Content
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Generating Domain Content for '${TOPIC}'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Modify domain generation to use specified topic
python -c "
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import domain generation modules
from src.domain_generation.config import config
from src.domain_generation.graphs import build_domain_graph
from src.domain_generation.utils import logger

# Configuration
domain_name = '${TOPIC}'
domain_description = f'Knowledge and information about {domain_name}'
output_dir = Path('${OUTPUT_DIR}')
output_dir.mkdir(exist_ok=True, parents=True)

logger.info('='*80)
logger.info('Domain Content Generation')
logger.info('='*80)
logger.info(f'Domain: {domain_name}')
logger.info(f'Description: {domain_description}')
logger.info(f'Model: {config.model_name}')
logger.info(f'Output: {output_dir}')
logger.info('='*80)

# Build and run domain graph
logger.info('Building domain generation graph...')
domain_graph = build_domain_graph()

logger.info('Starting domain generation...')
result = domain_graph.invoke({
    'name': domain_name,
    'description': domain_description,
})

domain = result['domain']

# Log results
logger.info('='*80)
logger.info('Generation Complete!')
logger.info('='*80)
logger.info(f'Topics: {len(domain.topics)}')
for topic in domain.topics:
    logger.info(f'  - {topic.name}')
logger.info(f'Books: {len(domain.books)}')
logger.info(f'Articles: {len(domain.articles)}')

# Count QA pairs
total_grounded_qa = sum(len(book.grounded_questions) for book in domain.books)
total_grounded_qa += sum(len(article.grounded_questions) for article in domain.articles)
logger.info(f'Total Grounded QA Pairs: {total_grounded_qa}')

# Save outputs
output_file = output_dir / 'domain.json'
domain_dict = domain.model_dump()
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(domain_dict, f, indent=2, ensure_ascii=False)

logger.success(f'✅ Saved domain JSON to {output_file}')
logger.info('='*80)
"

echo ""
echo "✅ Domain generation complete!"
echo ""

##############################################################################
# Step 2: Convert to HuggingFace Dataset Format
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Converting to HuggingFace Dataset Format"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m src.domain_generation.convert_to_dataset \
    "${OUTPUT_DIR}/domain.json" \
    --output-dir "${DATA_DIR}" \
    --dataset-name "${DATASET_NAME}" \
    --split-ratio 0.8

echo ""
echo "✅ Dataset conversion complete!"
echo ""

##############################################################################
# Step 3: Create Dataset Config Files
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Creating Dataset Configuration Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create config directory for domain datasets
CONFIG_DIR="configs/data/datasets"
mkdir -p "${CONFIG_DIR}"

# Create forget dataset config (QA format)
cat > "${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_forget.yaml" << EOF
DOMAIN_${DATASET_NAME}_forget:
  handler: QADataset
  args:
    hf_args:
      path: "${DATA_DIR}/${DATASET_NAME}/qa_dataset"
      split: "forget"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

echo "Created: ${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_forget.yaml"

# Create retain dataset config (QA format)
cat > "${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_retain.yaml" << EOF
DOMAIN_${DATASET_NAME}_retain:
  handler: QADataset
  args:
    hf_args:
      path: "${DATA_DIR}/${DATASET_NAME}/qa_dataset"
      split: "retain"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

echo "Created: ${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_retain.yaml"

echo ""
echo "✅ Dataset configuration files created!"
echo ""

##############################################################################
# Step 4: Create Experiment Config
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Creating Experiment Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create experiment config directory
EXPERIMENT_CONFIG_DIR="configs/experiment/unlearn/domain"
mkdir -p "${EXPERIMENT_CONFIG_DIR}"

# Create experiment config
cat > "${EXPERIMENT_CONFIG_DIR}/${DATASET_NAME}.yaml" << EOF
# @package _global_

# Domain Unlearning Experiment: ${TOPIC}
# Generated: ${TIMESTAMP}

defaults:
  - /model: ${MODEL}
  - /trainer: ${TRAINER}
  - /collator: DataCollatorForSupervisedDataset
  - _self_

# Model configuration
model:
  model_args:
    pretrained_model_name_or_path: meta-llama/Meta-${MODEL}

# Data configuration
data:
  anchor: forget
  forget:
    DOMAIN_${DATASET_NAME}_forget:
      handler: QADataset
      args:
        hf_args:
          path: "${DATA_DIR}/${DATASET_NAME}/qa_dataset"
          split: "forget"
        question_key: "question"
        answer_key: "answer"
        max_length: 512
  retain:
    DOMAIN_${DATASET_NAME}_retain:
      handler: QADataset
      args:
        hf_args:
          path: "${DATA_DIR}/${DATASET_NAME}/qa_dataset"
          split: "retain"
        question_key: "question"
        answer_key: "answer"
        max_length: 512

# Task name
task_name: ${RUN_NAME}

# Evaluation configuration (optional)
eval: null
retain_logs_path: null
EOF

echo "Created: ${EXPERIMENT_CONFIG_DIR}/${DATASET_NAME}.yaml"
echo ""
echo "✅ Experiment configuration created!"
echo ""

##############################################################################
# Step 5: Run Unlearning
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Running Unlearning with ${TRAINER}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Set master port for distributed training
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: ${MASTER_PORT}"
echo ""

# Run unlearning
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/domain/${DATASET_NAME} \
    trainer=${TRAINER} \
    task_name=${RUN_NAME} \
    model=${MODEL} \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.save_strategy=epoch \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=10 \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.gradient_checkpointing=true

echo ""
echo "✅ Unlearning complete!"
echo ""

##############################################################################
# Step 6: Save Run Summary
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Saving Run Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create run summary
cat > "${DATA_DIR}/run_summary.json" << EOF
{
  "topic": "${TOPIC}",
  "dataset_name": "${DATASET_NAME}",
  "run_name": "${RUN_NAME}",
  "timestamp": "${TIMESTAMP}",
  "model": "${MODEL}",
  "trainer": "${TRAINER}",
  "hyperparameters": {
    "num_epochs": ${NUM_EPOCHS},
    "learning_rate": ${LEARNING_RATE},
    "per_device_batch_size": ${PER_DEVICE_BATCH_SIZE},
    "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS}
  },
  "paths": {
    "domain_json": "${OUTPUT_DIR}/domain.json",
    "data_dir": "${DATA_DIR}",
    "qa_dataset": "${DATA_DIR}/${DATASET_NAME}/qa_dataset",
    "text_dataset": "${DATA_DIR}/${DATASET_NAME}/text_dataset",
    "model_checkpoint": "saves/unlearn/${RUN_NAME}",
    "experiment_config": "${EXPERIMENT_CONFIG_DIR}/${DATASET_NAME}.yaml"
  }
}
EOF

echo "Created: ${DATA_DIR}/run_summary.json"
echo ""

##############################################################################
# Final Summary
##############################################################################

echo "================================================================================================"
echo "Domain Unlearning Pipeline Complete! 🎉"
echo "================================================================================================"
echo ""
echo "Summary:"
echo "  Topic:                ${TOPIC}"
echo "  Dataset:              ${DATASET_NAME}"
echo "  Model:                ${MODEL}"
echo "  Trainer:              ${TRAINER}"
echo "  Run Name:             ${RUN_NAME}"
echo ""
echo "Generated Artifacts:"
echo "  📄 Domain JSON:       ${OUTPUT_DIR}/domain.json"
echo "  📦 QA Dataset:        ${DATA_DIR}/${DATASET_NAME}/qa_dataset"
echo "  📦 Text Dataset:      ${DATA_DIR}/${DATASET_NAME}/text_dataset"
echo "  🧠 Model Checkpoint:  saves/unlearn/${RUN_NAME}"
echo "  📋 Run Summary:       ${DATA_DIR}/run_summary.json"
echo ""
echo "Next Steps:"
echo "  1. Evaluate the unlearned model:"
echo "     python src/eval.py \\"
echo "       model=${MODEL} \\"
echo "       model.model_args.pretrained_model_name_or_path=saves/unlearn/${RUN_NAME} \\"
echo "       task_name=${RUN_NAME}_eval"
echo ""
echo "  2. Test the model with queries about '${TOPIC}' to verify unlearning"
echo ""
echo "  3. Compare with baseline model to measure forget quality"
echo ""
echo "================================================================================================"
