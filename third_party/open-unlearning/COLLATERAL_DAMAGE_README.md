# Pipeline de Desaprendizagem para Dano Colateral

Este diretório contém uma documentação para o pipeline customizado, projetado para executar experimentos de desaprendizagem focados em medir o "dano colateral" em um benchmark pré-existente. Diferente do pipeline de *Domain Unlearning*, este não gera novo conteúdo, mas utiliza o arquivo `data/datasets/collateral_damage_probes.jsonl`.

## 🚀 Início Rápido

O pipeline é orquestrado pelo script `run-collateral-damage-unlearning.sh`.

```bash
# Exemplo básico:
# Executa um experimento no grupo "michael_jordan", marcando 1 entidade para esquecer.
# Usa o modelo e o trainer padrão (Llama-3.2-1B-Instruct e GradAscent).
bash third_party/scripts/run-collateral-damage-unlearning.sh "michael_jordan" 1

# Exemplo avançado:
# Executa um experimento no grupo "lincoln", marcando 2 entidades para esquecer,
# usando um modelo e trainer customizados.
bash third_party/scripts/run-collateral-damage-unlearning.sh "lincoln" 2 "meta-llama/Meta-Llama-3-8B-Instruct" "NPO"
```

## 📋 Visão Geral

O objetivo deste pipeline é facilitar a avaliação do impacto que o desaprendizado de uma entidade (`E_t`) tem sobre o conhecimento do modelo a respeito de entidades relacionadas, mas distintas (`E_a`).

O processo consiste nos seguintes passos, automatizados pelos scripts:

1.  **Preparação do Dataset**: O script `third_party/src/collateral_damage/run_unlearning_experiment.py` lê o arquivo `collateral_damage_probes.jsonl`, filtra pelo `group_id` especificado, e divide as entidades do grupo em um conjunto `forget` e um `retain`.
2.  **Criação de Configurações**: Gera dinamicamente os arquivos de configuração `.yaml` que o framework `open-unlearning` precisa para localizar os dados e configurar o experimento.
3.  **Execução do Desaprendizado**: Invoca o script `third_party/src/train.py` para executar o algoritmo de desaprendizagem no modelo especificado.
4.  **Resumo**: Ao final, exibe um resumo dos artefatos gerados e sugere o comando para a etapa de avaliação.

## 🔧 Argumentos do Script

O script `run-collateral-damage-unlearning.sh` aceita os seguintes argumentos:

1.  `GROUP_ID` (Obrigatório): A categoria do seu benchmark que será o foco do experimento (ex: "michael_jordan", "lincoln", "apple").
2.  `NUM_FORGET` (Obrigatório): O número de entidades únicas dentro do `GROUP_ID` que serão colocadas no conjunto `forget`. As restantes irão para o `retain`.
3.  `MODEL` (Opcional): O nome do modelo do Hugging Face a ser usado. O padrão é `meta-llama/Llama-3.2-1B-Instruct` para ser compatível com GPUs de ~16GB.
4.  `TRAINER` (Opcional): O método de desaprendizagem a ser usado. O padrão é `GradAscent`.

### Hiperparâmetros de Treinamento

Para um controle mais refinado, você pode executar o script Python diretamente e passar os seguintes argumentos para ajustar os hiperparâmetros de treinamento:

*   `--learning_rate`: A taxa de aprendizado (padrão: `1e-5`).
*   `--num_train_epochs`: Número de épocas de treinamento (padrão: `5`).
*   `--per_device_train_batch_size`: Batch size por GPU (padrão: `4`).
*   `--gradient_accumulation_steps`: Passos de acumulação de gradiente (padrão: `4`).

**Exemplo de uso avançado (executando o script Python diretamente):**
```bash
python third_party/src/collateral_damage/run_unlearning_experiment.py \
  --group_id "lincoln" \
  --num_forget 2 \
  --model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --learning_rate 5e-6 \
  --num_train_epochs 3
```

## ⚙️ Otimização para Baixo Consumo de VRAM

Os scripts foram configurados para rodar em GPUs com 16GB de VRAM, utilizando as seguintes otimizações:

*   **Modelo Padrão Pequeno**: `Llama-3.2-1B-Instruct`.
*   **Batch Size Mínimo**: `per_device_train_batch_size` é definido como `1`.
*   **Acumulação de Gradiente**: `gradient_accumulation_steps` é `4` para simular um batch size maior sem o custo de memória.
*   **Gradient Checkpointing**: Ativado por padrão (`true`) para economizar uma quantidade significativa de VRAM.
*   **Otimizador Paginado**: O framework utiliza por padrão o `paged_adamw_32bit`, que descarrega o estado do otimizador para a RAM da CPU.

## 📊 Próximos Passos: Avaliação

Após a conclusão do script de desaprendizagem, um modelo será salvo em `saves/unlearn/<run_name>/`. O passo mais importante da sua pesquisa é avaliar este modelo para medir o dano colateral.

Use o comando sugerido no final da execução do script para iniciar a avaliação:

```bash
# Exemplo de comando de avaliação
python third_party/src/eval.py \
    experiment=eval/tofu/default \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/michael_jordan_1forget_20251206_120000 \
    task_name=michael_jordan_1forget_20251206_120000_eval
```

A análise dos resultados desta avaliação (especialmente a performance no conjunto `retain`) indicará a magnitude do dano colateral.
