# ML Model Evaluation Metrics Calculator

[Portugues](#portugues) | [English](#english)

---

## English

### Overview

A comprehensive toolkit for calculating and visualizing machine learning model evaluation metrics. Implements classification and regression metrics with cross-validation support, statistical significance testing, and automated report generation.

**DIO Lab Project** - Formacao Machine Learning Specialist

### Features

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- **Regression Metrics**: MSE, RMSE, MAE, R-squared, Adjusted R-squared
- **Confusion Matrix**: Visualization with normalized and absolute values
- **ROC Curves**: Multi-class ROC curve plotting with AUC calculation
- **Cross-Validation**: K-Fold, Stratified K-Fold with confidence intervals
- **Statistical Tests**: McNemar, Wilcoxon signed-rank for model comparison
- **Report Generation**: Automated HTML/PDF metric reports

### Tech Stack

- Python 3.10+
- Scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn
- Docker
- GitHub Actions CI/CD

### Project Structure

```
python-ml-evaluation-metrics/
|-- src/
|   |-- __init__.py
|   |-- metrics_calculator.py
|   |-- visualization.py
|   |-- cross_validation.py
|   |-- report_generator.py
|   |-- model_comparison.py
|-- tests/
|   |-- __init__.py
|   |-- test_metrics.py
|   |-- test_visualization.py
|   |-- test_cross_validation.py
|-- data/
|   |-- sample_predictions.csv
|-- .github/
|   |-- workflows/
|       |-- ci.yml
|-- Dockerfile
|-- requirements.txt
|-- README.md
|-- LICENSE
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/galafis/python-ml-evaluation-metrics.git
cd python-ml-evaluation-metrics

# Install dependencies
pip install -r requirements.txt

# Run the metrics calculator
python -m src.metrics_calculator

# Run tests
pytest tests/ -v
```

### Docker

```bash
docker build -t ml-evaluation-metrics .
docker run --rm ml-evaluation-metrics
```

### Usage Example

```python
from src.metrics_calculator import MetricsCalculator
from src.visualization import MetricsVisualizer

# Initialize calculator
calculator = MetricsCalculator()

# Calculate classification metrics
results = calculator.evaluate_classification(
    y_true=[0, 1, 1, 0, 1, 0, 1, 1],
    y_pred=[0, 1, 0, 0, 1, 1, 1, 1],
    y_prob=[0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.7, 0.95]
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
print(f"ROC-AUC: {results['roc_auc']:.4f}")

# Visualize confusion matrix
visualizer = MetricsVisualizer()
visualizer.plot_confusion_matrix(results['confusion_matrix'])
visualizer.plot_roc_curve(results['fpr'], results['tpr'], results['roc_auc'])
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Portugues

### Visao Geral

Um toolkit abrangente para calculo e visualizacao de metricas de avaliacao de modelos de Machine Learning. Implementa metricas de classificacao e regressao com suporte a validacao cruzada, testes de significancia estatistica e geracao automatizada de relatorios.

**Projeto Lab DIO** - Formacao Machine Learning Specialist

### Funcionalidades

- **Metricas de Classificacao**: Acuracia, Precisao, Recall, F1-Score, ROC-AUC, PR-AUC
- **Metricas de Regressao**: MSE, RMSE, MAE, R-quadrado, R-quadrado Ajustado
- **Matriz de Confusao**: Visualizacao com valores normalizados e absolutos
- **Curvas ROC**: Plotagem de curvas ROC multi-classe com calculo de AUC
- **Validacao Cruzada**: K-Fold, Stratified K-Fold com intervalos de confianca
- **Testes Estatisticos**: McNemar, Wilcoxon signed-rank para comparacao de modelos
- **Geracao de Relatorios**: Relatorios automatizados de metricas em HTML/PDF

### Tecnologias

- Python 3.10+
- Scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn
- Docker
- GitHub Actions CI/CD

### Inicio Rapido

```bash
git clone https://github.com/galafis/python-ml-evaluation-metrics.git
cd python-ml-evaluation-metrics
pip install -r requirements.txt
python -m src.metrics_calculator
```

### Licenca

Este projeto esta licenciado sob a Licenca MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
