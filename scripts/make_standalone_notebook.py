from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/Trabalho_LoveDA_Segmentacao_Autocontido.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


core_code = Path("src/loveda_segmentation.py").read_text(encoding="utf-8")

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (LoveDA)",
        "language": "python",
        "name": "loveda-tl",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

nb["cells"] = [
    md(
        """
        # Trabalho Pratico - Segmentacao de Estradas e Construcoes com Transfer Learning

        **Tema:** identificacao automatica de elementos em imagens de satelite utilizando Deep Learning.

        **Base de dados:** LoveDA.

        **Objetivo:** desenvolver uma solucao de segmentacao semantica usando Transfer Learning para identificar regioes de interesse em imagens de satelite.

        A entrada do modelo e uma imagem RGB. A saida e uma mascara segmentada, indicando a classe de cada pixel.
        """
    ),
    md(
        """
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

        O foco principal da analise sera nas classes **construcao**, **estrada** e **vegetacao**.
        """
    ),
    md(
        """
        ## Preparacao do ambiente

        Execute a celula abaixo apenas se estiver abrindo este notebook fora do ambiente preparado no projeto.
        """
    ),
    code(
        """
        # Opcional: instalar dependencias, se necessario.
        # %pip install -U torch torchvision segmentation-models-pytorch albumentations opencv-python-headless pillow matplotlib pandas tqdm
        """
    ),
    md(
        """
        ## Imports, configuracoes e funcoes auxiliares

        Esta celula contem todo o codigo necessario para carregar a base, preparar imagens/mascaras, treinar os modelos, calcular metricas e visualizar resultados.
        """
    ),
    code(core_code),
    md(
        """
        ## Configuracao do experimento

        A configuracao usa o subconjunto sugerido: 100 imagens de treino, 30 de validacao e 30 de teste, com imagens redimensionadas para **256 x 256**.
        """
    ),
    code(
        """
        from pathlib import Path

        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent

        cfg = TrainConfig(
            data_root=str(PROJECT_ROOT / "data" / "LoveDA"),
            image_size=256,
            train_count=100,
            val_count=30,
            test_count=30,
            batch_size=8,
            epochs=5,
            lr=1e-3,
            seed=42,
            num_workers=2,
        )

        cfg
        """
    ),
    md(
        """
        ## Organizacao da base

        A LoveDA deve estar em `data/LoveDA`. Este notebook procura automaticamente subpastas `images_png` e `masks_png`.

        Estruturas aceitas:

        - `data/LoveDA/Train/Rural/images_png/*.png`
        - `data/LoveDA/Train/Rural/masks_png/*.png`
        - `data/LoveDA/Train/Urban/images_png/*.png`
        - `data/LoveDA/Train/Urban/masks_png/*.png`
        - `data/LoveDA/Val/Rural/images_png/*.png`
        - `data/LoveDA/Val/Rural/masks_png/*.png`
        """
    ),
    code(
        """
        pairs = find_loveda_pairs(cfg.data_root)
        train_pairs, val_pairs, test_pairs = split_pairs(
            pairs,
            cfg.train_count,
            cfg.val_count,
            cfg.test_count,
            cfg.seed,
        )

        print(f"Total encontrado: {len(pairs)} pares imagem/mascara")
        print(f"Treino: {len(train_pairs)} | Validacao: {len(val_pairs)} | Teste: {len(test_pairs)}")
        print("Primeiro par:", train_pairs[0])
        """
    ),
    md(
        """
        ## Visualizacao dos dados

        Antes do treinamento, sao exibidos 5 exemplos contendo imagem original, mascara real e legenda das classes presentes.
        """
    ),
    code("show_dataset_examples(train_pairs, n=5)"),
    md(
        """
        ## Preparacao das imagens

        As imagens sao redimensionadas, normalizadas e convertidas para tensores. As mascaras sao convertidas para o formato adequado de segmentacao, com valores inteiros por pixel.
        """
    ),
    code(
        """
        sample_ds = LoveDADataset(train_pairs[:2], cfg.image_size, augment=True)
        x, y = sample_ds[0]

        print("Imagem:", x.shape, x.dtype, float(x.min()), float(x.max()))
        print("Mascara:", y.shape, y.dtype, sorted(y.unique().tolist()))
        """
    ),
    md(
        """
        ## Modelos com e sem Transfer Learning

        Serao comparados tres cenarios:

        1. **U-Net sem Transfer Learning:** encoder ResNet34 inicializado do zero.
        2. **U-Net com Transfer Learning:** encoder ResNet34 pre-treinado no ImageNet.
        3. **U-Net com Transfer Learning + Data Augmentation:** mesmo modelo anterior com aumento de dados no conjunto de treino.

        A funcao de perda e **Cross-Entropy**, o otimizador e **Adam** e a metrica principal e **IoU medio**.
        """
    ),
    code(
        """
        models, history_df, metrics_df, splits = run_experiments(
            cfg,
            output_dir=PROJECT_ROOT / "outputs",
        )

        history_df.tail()
        """
    ),
    md("## Curvas de treinamento"),
    code("plot_history(history_df)"),
    md("## Metricas no conjunto de teste"),
    code(
        """
        metrics_cols = [
            "model",
            "loss",
            "pixel_accuracy",
            "mean_iou",
            "iou_building",
            "iou_road",
            "iou_vegetation",
        ]

        metrics_df[metrics_cols].sort_values("mean_iou", ascending=False)
        """
    ),
    md(
        """
        ## Avaliacao visual no conjunto de teste

        Para 5 imagens, sao exibidas:

        1. imagem original;
        2. mascara real;
        3. mascara prevista pelo modelo;
        4. sobreposicao da mascara prevista na imagem original.
        """
    ),
    code(
        """
        best_model_name = metrics_df.sort_values("mean_iou", ascending=False).iloc[0]["model"]
        print("Melhor modelo pelo IoU medio:", best_model_name)

        show_predictions(models[best_model_name], test_pairs, cfg, n=5)
        """
    ),
    md(
        """
        ## Analise dos resultados

        Analise visualmente os resultados considerando:

        - se as estradas foram identificadas corretamente;
        - se as construcoes foram identificadas corretamente;
        - se a vegetacao foi separada das areas agricolas;
        - se agua foi confundida com sombra;
        - se solo exposto foi confundido com estrada;
        - se construcoes pequenas foram detectadas;
        - se estradas estreitas foram perdidas;
        - se a segmentacao ficou fragmentada ou continua.
        """
    ),
    md(
        """
        ## Questoes para o relatorio

        1. **O modelo conseguiu identificar bem as estradas?**  
           Resposta sugerida: compare o `iou_road` e as imagens previstas. Estradas largas tendem a aparecer melhor; estradas estreitas podem ficar interrompidas.

        2. **O modelo conseguiu identificar bem as construcoes?**  
           Resposta sugerida: use o `iou_building`. Construcoes grandes geralmente sao reconhecidas com mais facilidade que construcoes pequenas e isoladas.

        3. **Qual classe foi mais dificil de segmentar?**  
           Resposta sugerida: escolha a classe com menor IoU entre as metricas por classe e confirme visualmente.

        4. **O Transfer Learning melhorou o resultado?**  
           Resposta sugerida: compare `unet_scratch` com `unet_tl` na tabela de metricas.

        5. **O Data Augmentation melhorou o resultado?**  
           Resposta sugerida: compare `unet_tl` com `unet_tl_aug`.

        6. **Quais foram os principais erros observados?**  
           Resposta sugerida: cite confusoes como solo exposto x estrada, agricultura x vegetacao, sombra x agua e perda de objetos pequenos.

        7. **O resultado seria util para uma aplicacao real? Justifique.**  
           Resposta sugerida: para triagem ou apoio visual, sim, se as metricas forem razoaveis; para uso operacional critico, seria necessario treinar com mais dados, mais epocas e validacao em regioes diferentes.
        """
    ),
]

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK)
print(f"Notebook autocontido gerado em {NOTEBOOK}")
