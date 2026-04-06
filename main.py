import time

import torch

from src.evaluate import (
    evaluate_model,
    generate_all_plots,
    print_classification_report,
)
from src.model import MLP
from src.train import OPTIMIZERS, train_model
from src.utils import get_data_loaders


INPUT_SIZE = 784   # 28 x 28
OUTPUT_SIZE = 10   # 10 classes Fashion-MNIST


def get_hyperparameters():
    """Solicita hiperparametros ao usuario via terminal."""
    print("=" * 60)
    print("  CONFIGURACAO DA MLP — Fashion-MNIST")
    print("=" * 60)

    n_layers = int(input("\nNumero de camadas ocultas: "))
    print(f"[INPUT] -> {n_layers} camadas ocultas selecionadas")

    hidden_layers = []
    for i in range(1, n_layers + 1):
        n = int(input(f"  Neuronios na camada {i}: "))
        hidden_layers.append(n)
    print(f"[INPUT] -> Arquitetura das camadas ocultas: {hidden_layers}")

    lr = float(input("\nTaxa de aprendizado (ex: 0.01): "))
    print(f"[INPUT] -> Learning rate: {lr}")

    momentum = float(input("Taxa de momento / momentum (ex: 0.9): "))
    print(f"[INPUT] -> Momentum: {momentum}")

    opt_options = list(OPTIMIZERS.keys())
    print(f"\nOtimizadores disponiveis: {opt_options}")
    optimizer_name = input("Escolha o otimizador: ").strip().lower()
    print(f"[INPUT] -> Otimizador: {optimizer_name}")

    epochs = int(input("\nNumero de epocas: "))
    print(f"[INPUT] -> Epocas: {epochs}")

    params = {
        "hidden_layers": hidden_layers,
        "lr": lr,
        "momentum": momentum,
        "optimizer_name": optimizer_name,
        "epochs": epochs,
    }

    print(f"\n[INPUT] Resumo dos hiperparametros:")
    for k, v in params.items():
        print(f"[INPUT]   {k}: {v}")

    return params


def main():
    start_time = time.time()
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "MLP — Fashion-MNIST" + " " * 19 + "#")
    print("#" * 60 + "\n")

    params = get_hyperparameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SISTEMA] Dispositivo selecionado: {device}")
    if device.type == "cuda":
        print(f"[SISTEMA] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[SISTEMA] Memoria GPU: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print(f"[SISTEMA] Treinamento em CPU (pode ser mais lento)")

    print(f"\n[ETAPA 1/4] Carregando dataset Fashion-MNIST...")
    train_loader, val_loader, test_loader = get_data_loaders()

    print(f"[ETAPA 2/4] Construindo modelo MLP...")
    model = MLP(INPUT_SIZE, params["hidden_layers"], OUTPUT_SIZE).to(device)
    print(f"\n[MODELO] Arquitetura completa:")
    print(model)
    print()

    print(f"[ETAPA 3/4] Iniciando treinamento...")
    print("-" * 60)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=params["epochs"],
        lr=params["lr"],
        momentum=params["momentum"],
        optimizer_name=params["optimizer_name"],
        device=device,
    )

    print(f"\n[ETAPA 4/4] Avaliando no conjunto de teste...")
    y_true, y_pred = evaluate_model(model, test_loader, device)

    print_classification_report(y_true, y_pred)

    generate_all_plots(history, y_true, y_pred, "outputs")

    total_time = time.time() - start_time
    print(f"\n{'#'*60}")
    print(f"  CONCLUIDO!")
    print(f"  Tempo total: {total_time:.1f}s")
    print(f"  Graficos salvos em: outputs/")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
