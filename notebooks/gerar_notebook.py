"""Monta e executa notebooks/analise.ipynb (python notebooks/gerar_notebook.py)."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

celulas = [
    md(
        "# Previsão de falhas em máquinas industriais\n\n"
        "**Pergunta:** a partir das leituras de sensores de uma fresadora, dá para prever se a "
        "próxima operação vai falhar, a tempo de parar a máquina e fazer manutenção?\n\n"
        "Dataset: [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601) "
        "(S. Matzka, UCI, licença CC BY 4.0), com 10 mil leituras."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n\n"
        "sys.path.insert(0, str(Path.cwd().parent))\n\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n\n"
        "from falhas import dados, modelo\n\n"
        "plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})\n"
        "bruto = pd.read_csv(dados.ARQUIVO_PADRAO)\n"
        "df = dados.carregar()\n"
        "df.head()"
    ),
    md("## 1. Classes desbalanceadas\n\nSó cerca de 3% das leituras terminam em falha. "
       "Por isso a acurácia engana: um modelo que sempre diz \"não vai falhar\" acerta 97%."),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))\n"
        "df['falha'].value_counts().rename({0: 'sem falha', 1: 'falha'}).plot.bar(ax=axes[0], color=['#9aa3ae', '#f26b1d'])\n"
        "axes[0].set_title('Leituras por resultado'); axes[0].tick_params(axis='x', rotation=0)\n"
        "(df.groupby('tipo')['falha'].mean() * 100).reindex(['L', 'M', 'H']).plot.bar(ax=axes[1], color='#f26b1d')\n"
        "axes[1].set_title('Taxa de falha por qualidade do produto (%)'); axes[1].tick_params(axis='x', rotation=0)\n"
        "plt.tight_layout()\n"
        "print(f\"Taxa de falha: {df['falha'].mean():.2%}\")"
    ),
    md(
        "## 2. Tipos de falha (e por que eles ficam fora do modelo)\n\n"
        "O dataset informa **qual** falha aconteceu: desgaste da ferramenta (TWF), dissipação de calor (HDF), "
        "potência (PWF), sobrecarga (OSF) e aleatória (RNF). Essas colunas só existem **depois** da falha. "
        "Usá-las como entrada seria vazamento de dados, então elas servem apenas para entender o problema."
    ),
    code(
        "tipos = bruto[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().sort_values(ascending=False)\n"
        "tipos.index = ['dissipação de calor' if t == 'HDF' else 'sobrecarga' if t == 'OSF' else 'potência' if t == 'PWF'\n"
        "               else 'desgaste da ferramenta' if t == 'TWF' else 'aleatória' for t in tipos.index]\n"
        "tipos.plot.barh(color='#f26b1d', figsize=(6, 3), title='Ocorrências por tipo de falha').invert_yaxis()"
    ),
    md(
        "## 3. Features baseadas na física da máquina\n\n"
        "Além das leituras originais, foram criadas três variáveis ligadas aos mecanismos de falha:\n\n"
        "| Feature | Fórmula | Mecanismo |\n|---|---|---|\n"
        "| `delta_temp_k` | T processo − T ar | calor mal dissipado quando a diferença é pequena |\n"
        "| `potencia_w` | torque × rotação × 2π/60 | falha com potência muito baixa ou muito alta |\n"
        "| `esforco_ferramenta` | desgaste × torque | ferramenta gasta trabalhando com força |\n"
    ),
    code(
        "variaveis = ['rotacao_rpm', 'torque_nm', 'delta_temp_k', 'potencia_w', 'esforco_ferramenta', 'desgaste_min']\n"
        "fig, axes = plt.subplots(2, 3, figsize=(12, 6))\n"
        "for ax, coluna in zip(axes.flat, variaveis):\n"
        "    for classe, cor, nome in [(0, '#9aa3ae', 'sem falha'), (1, '#f26b1d', 'falha')]:\n"
        "        ax.hist(df.loc[df['falha'] == classe, coluna], bins=40, density=True, alpha=0.6, color=cor, label=nome)\n"
        "    ax.set_title(coluna)\n"
        "axes[0, 0].legend()\n"
        "plt.tight_layout()"
    ),
    md("As falhas se concentram nas pontas: potência muito baixa ou muito alta, esforço alto e "
       "pouca diferença de temperatura. É exatamente o que as features novas capturam."),
    code(
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "normais, falhas = df[df['falha'] == 0], df[df['falha'] == 1]\n"
        "ax.scatter(normais['rotacao_rpm'], normais['torque_nm'], s=4, alpha=0.25, color='#9aa3ae', label='sem falha')\n"
        "ax.scatter(falhas['rotacao_rpm'], falhas['torque_nm'], s=14, color='#f26b1d', label='falha')\n"
        "ax.set_xlabel('rotação (rpm)'); ax.set_ylabel('torque (Nm)')\n"
        "ax.set_title('Falhas aparecem nos extremos de potência (torque × rotação)')\n"
        "ax.legend();"
    ),
    md(
        "## 4. Modelagem\n\n"
        "- Separação **80/20 estratificada**. O conjunto de teste só é usado uma vez, no final\n"
        "- **3 candidatos:** regressão logística, random forest e gradient boosting, todos com peso maior para a classe rara\n"
        "- Comparação pela **PR-AUC** (área sob a curva precisão-recall) em validação cruzada de 5 dobras, porque ela é "
        "a métrica adequada para classes desbalanceadas\n"
        "- **Limiar de decisão** escolhido pelo **F2**: deixar passar uma falha custa mais que um alarme falso, então o recall "
        "pesa o dobro da precisão"
    ),
    code(
        "resultado = modelo.treinar(df)\n"
        "pd.DataFrame(resultado.comparacao).T.sort_values('pr_auc', ascending=False)"
    ),
    code(
        "print(f'Modelo escolhido: {resultado.nome}')\n"
        "print(f'Limiar de decisão: {resultado.limiar:.2f}')\n"
        "pd.Series({k: v for k, v in resultado.teste.items() if k != 'matriz_confusao'}, name='teste').to_frame()"
    ),
    code(
        "cm = resultado.teste['matriz_confusao']\n"
        "print(f\"Das {cm['vp'] + cm['fn']} falhas do teste, o modelo antecipou {cm['vp']} \"\n"
        "      f\"e gerou {cm['fp']} alarmes falsos em {cm['vn'] + cm['fp']} operações normais.\")"
    ),
    md(
        "## 5. Conclusões\n\n"
        "- O modelo é **muito melhor que o acaso**: PR-AUC perto de 0,88 contra 0,034 de um palpite aleatório\n"
        "- Com o limiar escolhido pelo F2, ele antecipa a grande maioria das falhas e gera poucos alarmes falsos\n"
        "- As features físicas (`delta_temp_k`, `esforco_ferramenta`, `potencia_w`) estão entre as mais importantes "
        "(veja `relatorio/importancia_features.png`)\n"
        "- As falhas aleatórias (RNF), por definição, não são previsíveis pelos sensores\n\n"
        "**Próximos passos:** calibração das probabilidades, um limiar por tipo de produto e um "
        "monitoramento de *drift* com os dados de produção."
    ),
]

notebook = nbformat.v4.new_notebook(cells=celulas)
notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
pasta = Path(__file__).resolve().parent
NotebookClient(notebook, timeout=600, resources={"metadata": {"path": str(pasta)}}).execute()
nbformat.write(notebook, pasta / "analise.ipynb")
print("notebook gerado:", pasta / "analise.ipynb")
