# Trabalho Pratico - Segmentacao LoveDA

Este projeto entrega um notebook executavel para segmentacao semantica de imagens de satelite da base LoveDA, comparando:

- U-Net sem Transfer Learning;
- U-Net com encoder ResNet34 pre-treinado;
- U-Net com ResNet34 pre-treinado + Data Augmentation.

## Como usar

1. Coloque a base LoveDA em `data/LoveDA` ou informe o caminho no notebook.

   Download oficial via Zenodo:

```bash
bash scripts/download_loveda.sh
```

2. Crie/atualize o ambiente:

```bash
uv sync --python 3.12
uv pip install --reinstall torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

3. Ative o ambiente:

```bash
source .venv/bin/activate
```

4. Abra o notebook:

```bash
jupyter lab notebooks/Trabalho_LoveDA_Segmentacao.ipynb
```

4. Execute as celulas em ordem.

O codigo procura automaticamente pastas chamadas `images_png` e `masks_png`, inclusive dentro de estruturas como `Train/Rural`, `Train/Urban`, `Val/Rural` e `Val/Urban`.

## Classes da atividade

| ID | Classe |
|---:|---|
| 0 | Ignorar / sem dado |
| 1 | Fundo |
| 2 | Construcao |
| 3 | Estrada |
| 4 | Agua |
| 5 | Solo exposto |
| 6 | Floresta / vegetacao |
| 7 | Agricultura |

Se a mascara vier no padrao LoveDA com classes `0..6`, o codigo remapeia automaticamente para o padrao do enunciado.
# dp-trabalho
# dp-trabalho
# dp-trabalho
