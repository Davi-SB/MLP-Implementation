# MLP Implementation — PyTorch

Implementacao de uma rede neural Multilayer Perceptron (MLP) configuravel com PyTorch, treinada e avaliada no dataset Fashion-MNIST.

## Requisitos

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

O programa solicitara interativamente:
- Numero de camadas ocultas
- Numero de neuronios em cada camada
- Taxa de aprendizado
- Taxa de momento (momentum)
- Algoritmo otimizador (SGD, Adam, RMSprop, Adagrad)
- Numero de epocas

## Dataset

**Fashion-MNIST** — 70.000 imagens 28x28 em escala de cinza, 10 classes de roupas/acessorios. Baixado automaticamente via `torchvision`.

## Resultados

Os graficos de desempenho sao salvos na pasta `outputs/`:
- Curva de loss (treino vs. validacao)
- Curva de acuracia (treino vs. validacao)
- Matriz de confusao
- Acuracia por classe
