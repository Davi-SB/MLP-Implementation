"""Executa experimentos sistematicos variando hiperparametros da MLP."""
import json
import os
import time

import matplotlib.pyplot as plt
import torch

from src.evaluate import evaluate_model, generate_all_plots, print_classification_report
from src.model import MLP
from src.train import train_model
from src.utils import get_data_loaders

INPUT_SIZE = 784
OUTPUT_SIZE = 10
EPOCHS = 10
EXPERIMENT_DIR = "outputs/experiments"


def run_single_experiment(
    idx, total, name, hidden_layers, lr, momentum, optimizer_name,
    train_loader, val_loader, test_loader, device,
):
    """Executa um unico experimento e retorna resultados."""
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENTO {idx}/{total}: {name}")
    print(f"  Camadas: {hidden_layers} | LR: {lr} | Mom: {momentum} | Opt: {optimizer_name}")
    print(f"{'#'*60}")

    print(f"[EXP {idx}] Construindo modelo...")
    model = MLP(INPUT_SIZE, hidden_layers, OUTPUT_SIZE).to(device)

    print(f"[EXP {idx}] Iniciando treinamento ({EPOCHS} epocas)...")
    exp_start = time.time()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=lr,
        momentum=momentum,
        optimizer_name=optimizer_name,
        device=device,
    )

    print(f"[EXP {idx}] Avaliando no conjunto de teste...")
    y_true, y_pred = evaluate_model(model, test_loader, device)
    test_acc = (y_true == y_pred).sum() / len(y_true)

    exp_dir = os.path.join(EXPERIMENT_DIR, name.replace(" ", "_"))
    print(f"[EXP {idx}] Gerando graficos em '{exp_dir}'...")
    generate_all_plots(history, y_true, y_pred, exp_dir)

    exp_time = time.time() - exp_start
    print(f"[EXP {idx}] Experimento '{name}' concluido em {exp_time:.1f}s")
    print(f"[EXP {idx}] Acuracia no teste: {test_acc:.4f} ({test_acc:.2%})")

    return {
        "name": name,
        "hidden_layers": hidden_layers,
        "lr": lr,
        "momentum": momentum,
        "optimizer": optimizer_name,
        "params": sum(p.numel() for p in model.parameters()),
        "test_accuracy": float(test_acc),
        "best_val_acc": max(history["val_acc"]),
        "final_train_loss": history["train_loss"][-1],
        "history": history,
        "time_seconds": exp_time,
    }


def plot_comparison(results, key, ylabel, title, filename):
    """Gera grafico comparativo entre experimentos."""
    print(f"[COMP] Gerando grafico comparativo: {title}...")
    plt.figure(figsize=(14, 6))

    for r in results:
        epochs = range(1, len(r["history"][key]) + 1)
        plt.plot(epochs, r["history"][key], "-o", label=r["name"], markersize=2)

    plt.title(title)
    plt.xlabel("Epoca")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(EXPERIMENT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[COMP] -> Salvo: {path}")


def plot_accuracy_bar(results, filename="accuracy_comparison.png"):
    """Grafico de barras comparando acuracia final de todos os experimentos."""
    print(f"[COMP] Gerando grafico de barras comparativo...")
    names = [r["name"] for r in results]
    accs = [r["test_accuracy"] for r in results]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(names, accs, color="steelblue")

    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{acc:.2%}", ha="center", va="bottom", fontsize=8,
        )

    plt.title("Comparacao de Acuracia no Teste")
    plt.ylabel("Acuracia")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(EXPERIMENT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[COMP] -> Salvo: {path}")


def main():
    total_start = time.time()

    print("\n" + "#" * 60)
    print("#" + " " * 12 + "EXPERIMENTOS SISTEMATICOS MLP" + " " * 15 + "#")
    print("#" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] Dispositivo: {device}")

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    print(f"[SETUP] Diretorio de saida: {EXPERIMENT_DIR}")

    print(f"\n[SETUP] Carregando Fashion-MNIST...")
    train_loader, val_loader, test_loader = get_data_loaders()

    experiments = [
        # --- Variando arquitetura (camadas pequenas) ---
        {"name": "1 camada [64]",           "hidden_layers": [64],            "lr": 0.001, "momentum": 0.9, "optimizer_name": "adam"},
        {"name": "2 camadas [128,64]",      "hidden_layers": [128, 64],       "lr": 0.001, "momentum": 0.9, "optimizer_name": "adam"},
        {"name": "3 camadas [256,128,64]",  "hidden_layers": [256, 128, 64],  "lr": 0.001, "momentum": 0.9, "optimizer_name": "adam"},

        # --- Variando otimizador ---
        {"name": "SGD",     "hidden_layers": [128, 64], "lr": 0.01,  "momentum": 0.9, "optimizer_name": "sgd"},
        {"name": "Adam",    "hidden_layers": [128, 64], "lr": 0.001, "momentum": 0.0, "optimizer_name": "adam"},
        {"name": "RMSprop", "hidden_layers": [128, 64], "lr": 0.001, "momentum": 0.0, "optimizer_name": "rmsprop"},
        {"name": "Adagrad", "hidden_layers": [128, 64], "lr": 0.01,  "momentum": 0.0, "optimizer_name": "adagrad"},

        # --- Variando learning rate (com Adam) ---
        {"name": "LR=0.0001", "hidden_layers": [128, 64], "lr": 0.0001, "momentum": 0.0, "optimizer_name": "adam"},
        {"name": "LR=0.001",  "hidden_layers": [128, 64], "lr": 0.001,  "momentum": 0.0, "optimizer_name": "adam"},
    ]

    total_exp = len(experiments)
    print(f"\n[SETUP] Total de experimentos a executar: {total_exp}")
    print(f"[SETUP] Epocas por experimento: {EPOCHS}")
    print(f"[SETUP] Configuracoes:")
    for i, exp in enumerate(experiments, 1):
        print(f"[SETUP]   {i}. {exp['name']} (layers={exp['hidden_layers']}, lr={exp['lr']}, opt={exp['optimizer_name']})")
    print()

    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n[PROGRESSO] >>>>  Experimento {i}/{total_exp}  <<<<")
        elapsed = time.time() - total_start
        if i > 1:
            avg_per_exp = elapsed / (i - 1)
            remaining = avg_per_exp * (total_exp - i + 1)
            print(f"[PROGRESSO] Tempo decorrido: {elapsed:.0f}s | ETA restante: ~{remaining:.0f}s")

        result = run_single_experiment(
            idx=i,
            total=total_exp,
            **exp,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )
        results.append(result)

        print(f"\n[PROGRESSO] Concluidos: {i}/{total_exp} ({i/total_exp:.0%})")
        print(f"[PROGRESSO] Resultados parciais:")
        for r in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
            print(f"[PROGRESSO]   {r['test_accuracy']:.2%}  |  {r['name']}  ({r['time_seconds']:.0f}s)")

    # --- Graficos comparativos ---
    print(f"\n{'='*60}")
    print(f"  GERANDO GRAFICOS COMPARATIVOS")
    print(f"{'='*60}")

    arch_results = results[:3]
    opt_results = results[3:7]
    lr_results = results[7:]

    plot_comparison(arch_results, "val_acc", "Acuracia", "Validacao: Efeito da Arquitetura", "comp_arch_acc.png")
    plot_comparison(arch_results, "val_loss", "Loss", "Validacao: Efeito da Arquitetura (Loss)", "comp_arch_loss.png")

    plot_comparison(opt_results, "val_acc", "Acuracia", "Validacao: Comparacao de Otimizadores", "comp_opt_acc.png")
    plot_comparison(opt_results, "val_loss", "Loss", "Validacao: Comparacao de Otimizadores (Loss)", "comp_opt_loss.png")

    plot_comparison(lr_results, "val_acc", "Acuracia", "Validacao: Efeito da Learning Rate", "comp_lr_acc.png")
    plot_comparison(lr_results, "val_loss", "Loss", "Validacao: Efeito da Learning Rate (Loss)", "comp_lr_loss.png")

    plot_accuracy_bar(results)

    # --- Salvar resumo ---
    summary = []
    for r in results:
        summary.append({
            "name": r["name"],
            "hidden_layers": r["hidden_layers"],
            "lr": r["lr"],
            "momentum": r["momentum"],
            "optimizer": r["optimizer"],
            "params": r["params"],
            "test_accuracy": r["test_accuracy"],
            "best_val_acc": r["best_val_acc"],
            "time_seconds": r["time_seconds"],
        })

    summary_path = os.path.join(EXPERIMENT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[RESUMO] Resultados salvos em: {summary_path}")

    total_time = time.time() - total_start

    print(f"\n{'#'*60}")
    print(f"  RESUMO FINAL DOS EXPERIMENTOS")
    print(f"{'#'*60}")
    print(f"\n  Ranking por acuracia no teste:\n")
    for rank, r in enumerate(sorted(results, key=lambda x: x["test_accuracy"], reverse=True), 1):
        print(
            f"  {rank}. {r['test_accuracy']:.2%}  |  {r['name']:25s}  |  "
            f"params: {r['params']:>7,}  |  {r['time_seconds']:.0f}s"
        )

    print(f"\n  Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Graficos salvos em: {EXPERIMENT_DIR}/")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
