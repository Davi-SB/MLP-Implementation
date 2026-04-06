import torch.nn as nn


class MLP(nn.Module):
    """Multilayer Perceptron com arquitetura configuravel."""

    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int):
        super().__init__()

        print(f"[MODELO] Construindo MLP...")
        print(f"[MODELO]   Entrada:         {input_size} neuronios")

        layers = []
        prev_size = input_size

        for i, h in enumerate(hidden_layers, 1):
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            print(f"[MODELO]   Camada oculta {i}: {prev_size} -> {h} + ReLU")
            prev_size = h

        layers.append(nn.Linear(prev_size, output_size))
        print(f"[MODELO]   Camada saida:    {prev_size} -> {output_size}")

        self.network = nn.Sequential(*layers)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[MODELO] Total de parametros: {total_params:,}")
        print(f"[MODELO] Modelo construido com sucesso!\n")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)
