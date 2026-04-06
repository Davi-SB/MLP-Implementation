import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import FASHION_MNIST_CLASSES


def evaluate_model(model, test_loader, device: torch.device):
    """Avalia o modelo no conjunto de teste e retorna labels reais e preditos."""
    print(f"\n[AVALIACAO] Iniciando avaliacao no conjunto de teste...")
    print(f"[AVALIACAO] Batches a processar: {len(test_loader)}")

    model.eval()
    all_labels = []
    all_preds = []
    start = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, 1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"[AVALIACAO] Batch {batch_idx}/{len(test_loader)} processado")

    elapsed = time.time() - start
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = (y_true == y_pred).sum() / len(y_true)
    print(f"[AVALIACAO] Concluida em {elapsed:.2f}s")
    print(f"[AVALIACAO] Amostras avaliadas: {len(y_true)}")
    print(f"[AVALIACAO] Acuracia geral: {accuracy:.4f} ({accuracy:.2%})")

    return y_true, y_pred


def print_classification_report(y_true, y_pred):
    """Imprime precision, recall e F1 por classe."""
    print(f"\n{'='*60}")
    print("  CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(
        y_true, y_pred, target_names=FASHION_MNIST_CLASSES,
    ))

    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print(f"[AVALIACAO] Acuracia por classe:")
    for cls_name, acc in zip(FASHION_MNIST_CLASSES, per_class_acc):
        bar = "#" * int(acc * 30)
        print(f"[AVALIACAO]   {cls_name:>12s}: {acc:.2%}  {bar}")


def plot_training_curves(history: dict, output_dir: str):
    """Gera graficos de loss e acuracia ao longo das epocas."""
    print(f"[GRAFICO] Gerando curvas de treinamento...")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Treino", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Validacao", markersize=3)
    ax1.set_title("Loss por Epoca")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Treino", markersize=3)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Validacao", markersize=3)
    ax2.set_title("Acuracia por Epoca")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Acuracia")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[GRAFICO] -> Salvo: {path}")


def plot_confusion_matrix(y_true, y_pred, output_dir: str):
    """Gera heatmap da matriz de confusao."""
    print(f"[GRAFICO] Gerando matriz de confusao...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=FASHION_MNIST_CLASSES,
        yticklabels=FASHION_MNIST_CLASSES,
    )
    plt.title("Matriz de Confusao")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[GRAFICO] -> Salvo: {path}")


def plot_class_accuracy(y_true, y_pred, output_dir: str):
    """Gera grafico de barras com acuracia por classe."""
    print(f"[GRAFICO] Gerando grafico de acuracia por classe...")
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(FASHION_MNIST_CLASSES, per_class_acc, color="steelblue")

    for bar, acc in zip(bars, per_class_acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.1%}", ha="center", va="bottom", fontsize=9,
        )

    plt.title("Acuracia por Classe")
    plt.xlabel("Classe")
    plt.ylabel("Acuracia")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "class_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[GRAFICO] -> Salvo: {path}")


def generate_all_plots(history, y_true, y_pred, output_dir: str):
    """Gera todos os graficos de avaliacao."""
    print(f"\n[GRAFICO] Gerando todos os graficos em '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    plot_training_curves(history, output_dir)
    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_class_accuracy(y_true, y_pred, output_dir)
    print(f"[GRAFICO] Todos os graficos salvos com sucesso!\n")
