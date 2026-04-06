import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_data_loaders(batch_size: int = 64, val_split: float = 0.1):
    """Carrega Fashion-MNIST e retorna DataLoaders de treino, validacao e teste."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    print(f"[DADOS] Diretorio de dados: {data_dir}")
    print(f"[DADOS] Baixando/carregando Fashion-MNIST (treino)...")
    full_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform,
    )
    print(f"[DADOS] -> {len(full_train)} amostras de treino carregadas")

    print(f"[DADOS] Baixando/carregando Fashion-MNIST (teste)...")
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform,
    )
    print(f"[DADOS] -> {len(test_dataset)} amostras de teste carregadas")

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    print(f"[DADOS] Dividindo treino em {train_size} treino + {val_size} validacao (split={val_split})")

    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"[DADOS] DataLoaders criados (batch_size={batch_size})")
    print(f"[DADOS]   Treino:    {len(train_loader)} batches")
    print(f"[DADOS]   Validacao: {len(val_loader)} batches")
    print(f"[DADOS]   Teste:     {len(test_loader)} batches")
    print(f"[DADOS] Classes: {FASHION_MNIST_CLASSES}")
    print(f"[DADOS] Dados prontos!\n")

    return train_loader, val_loader, test_loader
