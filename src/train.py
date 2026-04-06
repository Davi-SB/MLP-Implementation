import time

import torch
import torch.nn as nn


OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}


def _get_optimizer(model, name: str, lr: float, momentum: float):
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Otimizador '{name}' nao suportado. Opcoes: {list(OPTIMIZERS.keys())}"
        )

    opt_class = OPTIMIZERS[name]

    if name == "sgd":
        return opt_class(model.parameters(), lr=lr, momentum=momentum)
    return opt_class(model.parameters(), lr=lr)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """Executa uma epoca (treino se optimizer fornecido, senao validacao)."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)

    context = torch.no_grad() if not is_training else torch.enable_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if is_training and batch_idx % 100 == 0:
                running_acc = correct / total
                running_loss = total_loss / total
                print(
                    f"         batch {batch_idx:4d}/{n_batches} | "
                    f"loss: {running_loss:.4f} | acc: {running_acc:.4f}"
                )

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    momentum: float,
    optimizer_name: str,
    device: torch.device,
):
    """Treina o modelo e retorna o historico de metricas."""
    criterion = nn.CrossEntropyLoss()
    optimizer = _get_optimizer(model, optimizer_name, lr, momentum)

    print(f"[TREINO] Configuracao:")
    print(f"[TREINO]   Otimizador:      {optimizer_name.upper()}")
    print(f"[TREINO]   Learning rate:   {lr}")
    print(f"[TREINO]   Momentum:        {momentum}")
    print(f"[TREINO]   Epocas:          {epochs}")
    print(f"[TREINO]   Criterio:        CrossEntropyLoss")
    print(f"[TREINO]   Dispositivo:     {device}")
    print(f"[TREINO] {'='*55}")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        print(f"\n[TREINO] --- Epoca {epoch}/{epochs} ---")
        print(f"[TREINO] Treinando...")

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer,
        )

        print(f"[TREINO] Validando...")
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        eta = (elapsed / epoch) * (epochs - epoch)

        print(
            f"[TREINO] Epoca {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"Tempo: {epoch_time:.1f}s | ETA: {eta:.0f}s"
        )

        if epoch > 1:
            prev_val = history["val_loss"][-2]
            delta = val_loss - prev_val
            direction = "subiu" if delta > 0 else "desceu"
            print(f"[TREINO] Val loss {direction} {abs(delta):.4f} vs epoca anterior")

    total_time = time.time() - total_start
    best_val_acc = max(history["val_acc"])
    best_epoch = history["val_acc"].index(best_val_acc) + 1

    print(f"\n[TREINO] {'='*55}")
    print(f"[TREINO] Treinamento concluido em {total_time:.1f}s")
    print(f"[TREINO] Melhor val acc: {best_val_acc:.4f} (epoca {best_epoch})")
    print(f"[TREINO] Loss final treino: {history['train_loss'][-1]:.4f}")
    print(f"[TREINO] Loss final val:    {history['val_loss'][-1]:.4f}")

    return history
