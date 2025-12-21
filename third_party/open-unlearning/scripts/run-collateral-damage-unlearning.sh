#!/bin/bash

##############################################################################
# Pipeline de Desaprendizagem de Dano Colateral
#
# Este script orquestra o processo de desaprendizagem para o benchmark
# de dano colateral, utilizando o script Python correspondente.
#
# Uso:
#   bash third_party/scripts/run-collateral-damage-unlearning.sh <GROUP_ID> <NUM_FORGET> [MODEL] [TRAINER]
#
# Exemplo:
#   bash third_party/scripts/run-collateral-damage-unlearning.sh "michael_jordan" 1
#   bash third_party/scripts/run-collateral-damage-unlearning.sh "lincoln" 2 "meta-llama/Meta-Llama-3-8B-Instruct" "NPO"
##############################################################################

set -e # Sai imediatamente se um comando falhar

# --- Configuração ---
# O script python agora ajusta seu próprio diretório de trabalho para a raiz do projeto.

# Analisa argumentos da linha de comando
GROUP_ID="${1}"
NUM_FORGET="${2}"
MODEL="${3:-meta-llama/Llama-3.2-1B-Instruct}" # Valor padrão se não for fornecido
TRAINER="${4:-GradAscent}" # Valor padrão se não for fornecido

# Verifica se os argumentos obrigatórios foram fornecidos
if [ -z "${GROUP_ID}" ] || [ -z "${NUM_FORGET}" ]; then
  echo "Uso: bash third_party/scripts/run-collateral-damage-unlearning.sh <GROUP_ID> <NUM_FORGET> [MODEL] [TRAINER]"
  echo "Exemplo: bash third_party/scripts/run-collateral-damage-unlearning.sh \"michael_jordan\" 1"
  exit 1
fi

echo "==============================================================================================="
echo "Iniciando Pipeline de Desaprendizagem de Dano Colateral"
echo "==============================================================================================="
echo "  Grupo ID:             ${GROUP_ID}"
echo "  Número a Esquecer:    ${NUM_FORGET}"
echo "  Modelo:               ${MODEL}"
echo "  Trainer:              ${TRAINER}"
echo "==============================================================================================="
echo ""

# Define a porta master para treinamento distribuído
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Porta Master (MASTER_PORT): ${MASTER_PORT}"
echo ""

# Executa o script Python
# O próprio script Python ajustará seu diretório de trabalho.
python third_party/src/collateral_damage/run_unlearning_experiment.py \
  --group_id "${GROUP_ID}" \
  --num_forget "${NUM_FORGET}" \
  --model "${MODEL}" \
  --trainer "${TRAINER}"

echo ""
echo "==============================================================================================="
echo "Pipeline de Desaprendizagem de Dano Colateral Concluído! 🎉"
echo "==============================================================================================="
