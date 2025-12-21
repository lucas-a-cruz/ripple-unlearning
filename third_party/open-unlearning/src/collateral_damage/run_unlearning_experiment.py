import argparse
import json
import os
import random
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Adiciona o diretório src do third_party ao path para importar o QADataset
# e outros componentes necessários do open-unlearning.
# Isso torna o script executável de qualquer lugar.
third_party_src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(third_party_src_path))

from datasets import Dataset, DatasetDict


def get_free_port():
    """Encontra e retorna uma porta TCP livre."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def prepare_dataset(
    input_file: Path,
    output_dir: Path,
    group_id: str,
    num_forget_entities: int,
):
    """
    Carrega os dados do benchmark, filtra por um group_id específico,
    divide em conjuntos de 'forget' e 'retain' e salva como um dataset Hugging Face.

    Args:
        input_file: Caminho para o arquivo .jsonl do benchmark.
        output_dir: Diretório para salvar o dataset processado.
        group_id: O group_id para isolar no experimento.
        num_forget_entities: O número de entidades a serem colocadas no conjunto 'forget'.
    """
    print(f"📄 Lendo o arquivo de benchmark: {input_file}")
    probes = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            probes.append(json.loads(line))

    # Filtra as probes para o group_id especificado
    group_probes = [p for p in probes if p.get("group_id") == group_id]
    if not group_probes:
        print(f"❌ Erro: Nenhum dado encontrado para o group_id '{group_id}'. Abortando.")
        sys.exit(1)

    print(f"🔬 Encontradas {len(group_probes)} probes para o grupo '{group_id}'.")

    # Identifica entidades únicas dentro do grupo
    entities = sorted(list({p["entity_id"] for p in group_probes}))
    if len(entities) <= num_forget_entities:
        print(
            f"❌ Erro: O número de entidades para esquecer ({num_forget_entities}) "
            f"deve ser menor que o número total de entidades no grupo ({len(entities)})."
        )
        sys.exit(1)

    # Seleciona aleatoriamente as entidades para o conjunto 'forget'
    random.shuffle(entities)
    forget_entities = set(entities[:num_forget_entities])
    retain_entities = set(entities[num_forget_entities:])

    print(f"🎯 Entidades a serem esquecidas ({len(forget_entities)}): {forget_entities}")
    print(f"🛡️ Entidades a serem retidas ({len(retain_entities)}): {retain_entities}")

    # Divide as probes nos conjuntos de forget e retain
    forget_probes = [p for p in group_probes if p["entity_id"] in forget_entities]
    retain_probes = [p for p in group_probes if p["entity_id"] in retain_entities]

    print(f"📊 Divisão final: {len(forget_probes)} probes para 'forget', {len(retain_probes)} probes para 'retain'.")

    # Cria um único DatasetDict com splits 'forget' e 'retain'
    dataset_dict = DatasetDict({
        "forget": Dataset.from_list(forget_probes),
        "retain": Dataset.from_list(retain_probes),
    })

    # Salva o DatasetDict em um único diretório
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"💾 DatasetDict salvo em: {output_dir}")
    
    return output_dir


def create_config_files(
    dataset_path: Path,
    dataset_name: str,
    model: str,
    trainer: str,
    run_name: str,
):
    """
    Cria os arquivos de configuração YAML necessários para o framework open-unlearning.
    """
    print("✍️  Criando arquivos de configuração YAML...")

    # Os caminhos são relativos à raiz do projeto, onde o script será executado
    project_root = Path.cwd()
    config_root = project_root / "third_party/configs"
    dataset_config_dir = config_root / "data/datasets"
    experiment_config_dir = config_root / "experiment/unlearn/collateral_damage"
    dataset_config_dir.mkdir(exist_ok=True, parents=True)
    experiment_config_dir.mkdir(exist_ok=True, parents=True)

    # O caminho do dataset agora aponta para o diretório do DatasetDict
    relative_dataset_path = dataset_path.relative_to(project_root)

    # --- Configuração do Dataset 'Forget' ---
    forget_config_path = dataset_config_dir / f"collateral_damage_{dataset_name}_forget.yaml"
    forget_config_content = f"""
# Autogerado por run_unlearning_experiment.py
collateral_damage_{dataset_name}_forget:
  handler: QADataset
  args:
    hf_args:
      path: "{relative_dataset_path.as_posix()}"
      split: "forget"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
"""
    forget_config_path.write_text(forget_config_content)
    print(f"   -> Criado: {forget_config_path}")

    # --- Configuração do Dataset 'Retain' ---
    retain_config_path = dataset_config_dir / f"collateral_damage_{dataset_name}_retain.yaml"
    retain_config_content = f"""
# Autogerado por run_unlearning_experiment.py
collateral_damage_{dataset_name}_retain:
  handler: QADataset
  args:
    hf_args:
      path: "{relative_dataset_path.as_posix()}"
      split: "retain"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
"""
    retain_config_path.write_text(retain_config_content)
    print(f"   -> Criado: {retain_config_path}")

    # --- Configuração do Experimento ---
    exp_config_path = experiment_config_dir / f"{dataset_name}.yaml"
    model_config_name = model.split("/")[-1]
    
    exp_config_content = f"""
# @package _global_

# Experimento de Desaprendizagem de Dano Colateral: {dataset_name}
# Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

defaults:
  - override /model: {model_config_name}
  - override /trainer: {trainer}
  - override /collator: DataCollatorForSupervisedDataset
  - override /eval: null
  - override /data/datasets@data.forget: null
  - override /data/datasets@data.retain: null
  - _self_

# Configuração do modelo
model:
  model_args:
    pretrained_model_name_or_path: {model}

# Configuração dos dados
data:
  anchor: forget
  forget:
    collateral_damage_{dataset_name}_forget:
      handler: QADataset
      args:
        hf_args:
          path: "{relative_dataset_path.as_posix()}"
          split: "forget"
        question_key: "question"
        answer_key: "answer"
        max_length: 512
  retain:
    collateral_damage_{dataset_name}_retain:
      handler: QADataset
      args:
        hf_args:
          path: "{relative_dataset_path.as_posix()}"
          split: "retain"
        question_key: "question"
        answer_key: "answer"
        max_length: 512

# Nome da tarefa
task_name: {run_name}

# Evaluation configuration (optional)
eval: null
retain_logs_path: null
"""
    exp_config_path.write_text(exp_config_content)
    print(f"   -> Criado: {exp_config_path}")
    print("✅ Arquivos de configuração criados!")
    return exp_config_path


def run_unlearning(
    experiment_config_path: Path,
    trainer: str,
    model: str,
    run_name: str,
    learning_rate: float,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    use_accelerate: bool,
):
    """
    Executa o script de treinamento de desaprendizagem.
    """
    print("\n" + "="*80)
    print(f"🚀 Iniciando o processo de desaprendizagem com o trainer '{trainer}'...")
    print("="*80)

    train_script_path = Path("third_party/src/train.py")

    # Garante que third_party/src esteja em PYTHONPATH para o subprocesso
    current_python_path = os.environ.get("PYTHONPATH", "")
    # O Path.resolve() garante que o caminho é absoluto, importante para PYTHONPATH
    new_python_path = str(Path("third_party/src").resolve())
    if current_python_path:
        new_python_path = f"{new_python_path}{os.pathsep}{current_python_path}"

    env = os.environ.copy()
    env["PYTHONPATH"] = new_python_path
    env["CUDA_VISIBLE_DEVICES"] = "0,1"

    # O config_name é relativo ao config_path.
    config_name = experiment_config_path.relative_to(Path.cwd() / "third_party/configs/experiment").as_posix()
    print(f'CONFIG PATH = {config_name}')

    # Constrói os argumentos para o script de treinamento
    train_args = [
        str(train_script_path),
        f"--config-name=unlearn.yaml",
        f"experiment={config_name}",
        f"trainer={trainer}",
        f"task_name={run_name}",
        f"model={model.split('/')[-1]}",
        f"trainer.args.learning_rate={learning_rate}",
        f"trainer.args.num_train_epochs={num_train_epochs}",
        f"trainer.args.per_device_train_batch_size={per_device_train_batch_size}",
        f"trainer.args.gradient_accumulation_steps={gradient_accumulation_steps}",
        "trainer.args.save_strategy=epoch",
        "trainer.args.eval_strategy=no",
        "trainer.args.logging_steps=10",
        "trainer.args.ddp_find_unused_parameters=false",
        "trainer.args.gradient_checkpointing=true",
    ]

    if use_accelerate:
        print("🔧 Configurando para execução Multi-GPU com Accelerate e DeepSpeed.")
        port = get_free_port()
        print(f"   -> Porta Principal (MASTER_PORT): {port}")

        # O executável agora é 'accelerate'. Ele gerencia a execução do script.
        command = [
            "accelerate", "launch",
            "--config_file", "third_party/configs/accelerate/default_config.yaml",
            "--main_process_port", str(port),
            *train_args
        ]
    else:
        print("🔧 Configurando para execução padrão (sem Accelerate).")
        command = [sys.executable, *train_args]

    print(f"\nComando de execução:\n{' '.join(command)}\n")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1, env=env)

        for line in iter(process.stdout.readline, ''):
            print(line, end='')

        process.wait()

        if process.returncode != 0:
            print(f"❌ O processo de desaprendizagem falhou com o código de saída {process.returncode}.")
            sys.exit(process.returncode)

        print("\n✅ Processo de desaprendizagem concluído com sucesso!")

    except FileNotFoundError:
        print(f"❌ Erro: O executável '{command[0]}' ou o script de treinamento não foi encontrado.")
        print("   Certifique-se de que o ambiente está configurado corretamente e que os caminhos estão corretos.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Uma exceção ocorreu durante a execução do desaprendizagem: {e}")
        sys.exit(1)


def save_run_summary(
    args,
    dataset_name: str,
    run_name: str,
    timestamp: str,
    data_run_dir: Path,
    dataset_path: Path,
    exp_config_path: Path,
):
    """
    Salva um resumo da execução do experimento em um arquivo JSON.
    """
    print("\n" + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Step: Saving Run Summary")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")

    summary_data = {
        "group_id": args.group_id,
        "num_forget": args.num_forget,
        "model": args.model,
        "trainer": args.trainer,
        "dataset_name": dataset_name,
        "run_name": run_name,
        "timestamp": timestamp,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
        "paths": {
            "input_file": str(args.input_file),
            "processed_dataset_dir": str(dataset_path),
            "experiment_config_file": str(exp_config_path),
            "model_checkpoint_dir": f"saves/unlearn/{run_name}/",
            "run_summary_file": str(data_run_dir / "run_summary.json"),
        }
    }

    summary_file_path = data_run_dir / "run_summary.json"
    summary_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"Created: {summary_file_path}")
    print("✅ Run summary saved!")
    print("")


def main():
    parser = argparse.ArgumentParser(description="Executa um experimento de desaprendizagem de dano colateral.")
    parser.add_argument(
        "--group_id",
        type=str,
        required=True,
        help="O 'group_id' do benchmark para focar no experimento (ex: 'michael_jordan')."
    )
    parser.add_argument(
        "--num_forget",
        type=int,
        required=True,
        help="O número de entidades a serem usadas no conjunto 'forget'."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="O nome do modelo do Hugging Face a ser usado."
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="GradAscent",
        help="O método de desaprendizagem (trainer) a ser usado (ex: GradAscent, NPO)."
    )
    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Se definido, usa o Hugging Face Accelerate para executar o treinamento distribuído."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default="data/datasets/collateral_damage_probes.jsonl",
        help="Caminho para o arquivo .jsonl do benchmark."
    )
    # Hiperparâmetros de Treinamento
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Taxa de aprendizado para o treinamento.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Número de épocas de treinamento.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size por dispositivo.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Passos para acumulação de gradiente.")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"{args.group_id.replace(' ', '_')}_{args.num_forget}"
    run_name = f"{dataset_name}_{timestamp}"
    
    project_root = Path.cwd()
    data_run_dir = project_root / f"data/run/{dataset_name}/{timestamp}"

    print("="*80)
    print("🚀 Iniciando Experimento de Desaprendizagem de Dano Colateral 🚀")
    print("="*80)
    print(f"🆔 Grupo:               {args.group_id}")
    print(f"🔢 Entidades a Esquecer: {args.num_forget}")
    print(f"🤖 Modelo:              {args.model}")
    print(f"🎓 Trainer:             {args.trainer}")
    print(f"🚀 Accelerate:          {'Sim' if args.accelerate else 'Não'}")
    print(f"🏷️ Nome da Execução:    {run_name}")
    print(f"📂 Diretório de Dados:  {data_run_dir}")
    print("="*80)

    # --- Passo 1: Preparar o Dataset ---
    dataset_path = prepare_dataset(
        project_root / args.input_file,
        data_run_dir / "dataset",
        args.group_id,
        args.num_forget,
    )

    # --- Passo 2: Criar Arquivos de Configuração ---
    exp_config_path = create_config_files(
        dataset_path,
        dataset_name,
        args.model,
        args.trainer,
        run_name,
    )

    # --- Passo 3: Executar o Desaprendizagem ---
    run_unlearning(
        exp_config_path,
        args.trainer,
        args.model,
        run_name,
        args.learning_rate,
        args.num_train_epochs,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.accelerate,
    )

    # --- Passo 4: Salvar Resumo da Execução ---
    save_run_summary(
        args,
        dataset_name,
        run_name,
        timestamp,
        data_run_dir,
        dataset_path,
        exp_config_path,
    )

    # --- Resumo Final ---
    model_config_name = args.model.split('/')[-1]
    print("\n" + "="*80)
    print("🎉 Experimento Concluído! 🎉")
    print("================================================================================================")
    print("Resumo dos Artefatos Gerados:")
    print(f"  📦 Dataset Processado: {dataset_path}")
    print(f"  🧠 Checkpoint do Modelo: saves/unlearn/{run_name}/")
    print(f"  📋 Config do Experimento: {exp_config_path}")
    print(f"  📄 Sumário da Execução: {data_run_dir / 'run_summary.json'}")
    print("\nPróximos Passos Sugeridos:")
    print("  1. Avalie o modelo desaprendido para medir o dano colateral:")
    print(f"     python third_party/src/eval.py \
       experiment=eval/tofu/default \
       model={model_config_name} \
       model.model_args.pretrained_model_name_or_path=saves/unlearn/{run_name} \
       task_name={run_name}_eval")
    print("="*80)


if __name__ == "__main__":
    # Garante que o script seja executado a partir da raiz do projeto.
    # Resolve o caminho absoluto do script e sobe na árvore de diretórios.
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    os.chdir(project_root)
    
    main()
    