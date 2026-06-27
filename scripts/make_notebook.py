from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/Trabalho_LoveDA_Segmentacao.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


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

        **Base:** LoveDA.

        **Objetivo:** treinar modelos de segmentacao semantica que recebem uma imagem RGB e retornam uma mascara com uma classe por pixel.
        """
    ),
    md(
        """
        ## Classes utilizadas

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

        O foco principal da analise e nas classes **construcao**, **estrada** e **vegetacao**.
        """
    ),
    md(
        """
        ## Preparacao do ambiente

        Se este notebook for aberto fora do ambiente criado no projeto, execute a instalacao abaixo uma vez.
        """
    ),
    code(
        """
        # Opcional: execute somente se as bibliotecas nao estiverem instaladas.
        # %pip install -U torch torchvision segmentation-models-pytorch albumentations opencv-python-headless pillow matplotlib pandas tqdm
        """
    ),
    code(
        """
        from pathlib import Path
        import sys

        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / "src").exists() and (PROJECT_ROOT.parent / "src").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent
        sys.path.append(str(PROJECT_ROOT / "src"))

        from loveda_segmentation import *

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

        Coloque a LoveDA em `data/LoveDA`. O codigo procura automaticamente pares em subpastas `images_png` e `masks_png`.

        Exemplos aceitos:

        - `data/LoveDA/Train/Rural/images_png/*.png`
        - `data/LoveDA/Train/Rural/masks_png/*.png`
        - `data/LoveDA/Train/Urban/images_png/*.png`
        - `data/LoveDA/Train/Urban/masks_png/*.png`
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

        print(f"Total encontrado: {len(pairs)} pares")
        print(f"Treino: {len(train_pairs)} | Validacao: {len(val_pairs)} | Teste: {len(test_pairs)}")
        print("Primeiro par:", train_pairs[0])
        """
    ),
    md(
        """
        ## Visualizacao dos dados

        Abaixo sao exibidos 5 exemplos com imagem original, mascara real e legenda das classes presentes.
        """
    ),
    code(
        """
        show_dataset_examples(train_pairs, n=5)
        """
    ),
    md(
        """
        ## Preparacao das imagens

        As imagens sao redimensionadas para **256 x 256**, normalizadas com media/desvio do ImageNet e convertidas para tensores.

        As mascaras sao redimensionadas com interpolacao de vizinho mais proximo, convertidas para `long` e remapeadas para o padrao da atividade quando necessario.
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
        ## Modelos comparados

        1. **U-Net sem Transfer Learning:** encoder ResNet34 inicializado do zero.
        2. **U-Net com Transfer Learning:** encoder ResNet34 pre-treinado no ImageNet.
        3. **U-Net com Transfer Learning + Data Augmentation:** mesmo modelo anterior com aumento de dados no treino.

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
    md("## Curvas de treino"),
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
        ## Avaliacao visual no teste

        Para pelo menos 5 imagens sao exibidas:

        1. imagem original;
        2. mascara real;
        3. mascara prevista;
        4. sobreposicao da mascara prevista na imagem.
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

        Preencha apos executar o notebook, observando as imagens e a tabela de metricas:

        - **Estradas:** avaliar se os trechos principais ficaram continuos, se vias estreitas foram perdidas e se houve confusao com solo exposto.
        - **Construcoes:** verificar se construcoes grandes foram detectadas e se construcoes pequenas ficaram ausentes ou fragmentadas.
        - **Vegetacao:** observar se floresta/vegetacao foi separada de agricultura, principalmente em regioes de textura semelhante.
        - **Agua:** verificar se corpos d'agua foram confundidos com sombra.
        - **Solo exposto:** observar confusao com estrada e areas urbanas claras.
        """
    ),
    md(
        """
        ## Questoes do relatorio

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
print(f"Notebook gerado em {NOTEBOOK}")
